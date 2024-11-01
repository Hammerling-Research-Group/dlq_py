import os
import pandas as pd
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import pickle
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Helper function for spike detection (converted from R)
def find_spikes(times, obs, going_up_threshold=0.25, return_threshold=5, amp_threshold=1,
                cont_diff_threshold=0.25, cont_diff_num=10, make_plot=False):
    # Convert obs to numpy array if it's a pandas Series
    if isinstance(obs, pd.Series):
        obs = obs.to_numpy()

    # Ensure obs is a numpy array
    obs = np.array(obs)

    # Convert obs to float type, replacing non-numeric values with NaN
    obs = np.array(obs, dtype=float)

    # Check if obs is empty or all NaN
    if len(obs) == 0 or np.all(np.isnan(obs)):
        return pd.DataFrame({'time': times, 'events': np.full(len(times), np.nan)})

    events = np.full(len(obs), np.nan)
    count = 0
    in_event = False

    start_ind = np.min(np.where(~np.isnan(obs))[0]) + 1
    last_ob = False
    background = np.nan

    for i in range(start_ind, len(obs)):
        if i == len(obs) - 1:
            last_ob = True

        if np.isnan(obs[i]) and not np.isnan(obs[i - 1]):
            last_ind = i - 1
            continue
        elif np.isnan(obs[i]) and np.isnan(obs[i - 1]):
            continue
        elif not np.isnan(obs[i]) and np.isnan(obs[i - 1]):
            current_ind = i
        else:
            current_ind = i
            last_ind = i - 1

        if not in_event:
            current_diff = obs[current_ind] - obs[last_ind]
            threshold_to_use = max(going_up_threshold, amp_threshold) if last_ob else going_up_threshold

            if current_diff > threshold_to_use:
                in_event = True
                count += 1
                event_obs = [obs[current_ind]]
                events[current_ind] = count
                background = obs[last_ind]
        else:
            current_max = max(event_obs) - background
            current_ob = obs[current_ind] - background

            if (current_ob < 2 * background and current_ob < return_threshold * current_max / 100) or last_ob:
                in_event = False
                event_seq = range(int(np.min(np.where(events == count)[0])),
                                  int(np.max(np.where(events == count)[0])) + 1)
                events[event_seq] = count

                if last_ob:
                    event_size = max(event_obs) - background
                else:
                    event_size = max(event_obs) - np.mean([background, obs[current_ind]])

                if event_size < amp_threshold:
                    events[events == count] = np.nan
                    count -= 1
            else:
                event_obs.append(obs[current_ind])
                events[i] = count

                if len(event_obs) > cont_diff_num:
                    window_start = len(event_obs) - cont_diff_num
                    obs_in_window = event_obs[window_start:]

                    if all(abs(np.diff(obs_in_window)) < cont_diff_threshold):
                        in_event = False
                        event_seq = range(int(np.min(np.where(events == count)[0])),
                                          int(np.max(np.where(events == count)[0])) + 1)
                        events[event_seq] = count
                        event_size = max(event_obs) - np.mean([background, obs[current_ind]])

                        if event_size < amp_threshold:
                            events[events == count] = np.nan
                            count -= 1
                        else:
                            events[current_ind - cont_diff_num + 1:current_ind + 1] = np.nan

    filtered_events = pd.DataFrame({'time': times, 'events': events})

    if make_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(times, obs, linewidth=2)
        plt.xlabel('')
        plt.ylabel('Methane [ppm]')
        plt.ylim(0, np.nanmax(obs))

        if not np.all(np.isnan(events)):
            event_nums = np.unique(events[~np.isnan(events)])
            colors = plt.cm.rainbow(np.linspace(0, 1, len(event_nums)))

            for i, event_num in enumerate(event_nums):
                this_spike = np.where(events == event_num)[0]
                plt.scatter(times.iloc[this_spike], obs[this_spike], color=colors[i])
                plt.plot(times.iloc[this_spike], obs[this_spike], color=colors[i], linewidth=2)

        plt.show()

    return filtered_events


def remove_background(obs, times, gap_time=5, amp_threshold=0.75):
    # Convert obs to pandas DataFrame if it's not already
    if not isinstance(obs, pd.DataFrame):
        obs = pd.DataFrame(obs)

    # Convert columns to numeric where possible
    for col in obs.columns:
        obs[col] = pd.to_numeric(obs[col], errors='coerce')

    # Identify numeric columns
    numeric_columns = obs.select_dtypes(include=[np.number]).columns

    # Interpolate NA values only for numeric columns
    obs_interpolated = obs.copy()
    obs_interpolated[numeric_columns] = obs_interpolated[numeric_columns].interpolate(method='linear', axis=0,
                                                                                      limit_direction='both')

    to_use = obs_interpolated.columns[obs_interpolated.notna().any()]

    for j in to_use:
        this_raw_obs = obs_interpolated[j]
        to_keep = ~this_raw_obs.isna()
        trimmed_obs = this_raw_obs[to_keep]
        trimmed_times = times[to_keep]

        spikes = find_spikes(trimmed_times, trimmed_obs.values, amp_threshold=amp_threshold, make_plot=False)

        # Ensure spikes and trimmed_obs have the same index
        spikes.index = trimmed_obs.index

        # Add points immediately before and after the spike to the spike mask
        event_nums = np.unique(spikes['events'].dropna())
        for i in event_nums:
            event_mask = spikes['events'] == i
            event_indices = np.where(event_mask)[0]
            if len(event_indices) > 0:
                min_idx = event_indices.min()
                max_idx = event_indices.max()
                start_idx = max(min_idx - 1, 0)
                end_idx = min(max_idx + 1, len(spikes) - 1)
                spikes.iloc[start_idx:end_idx + 1, spikes.columns.get_loc('events')] = i

        # Combine events separated by less than gap_time
        if len(event_nums) > 1:
            for i in range(1, len(event_nums)):
                this_spike = spikes['events'] == event_nums[i]
                previous_spike = spikes['events'] == event_nums[i - 1]

                this_spike_start_time = spikes.index[this_spike].min()
                previous_spike_end_time = spikes.index[previous_spike].max()

                time_diff = (this_spike_start_time - previous_spike_end_time).total_seconds() / 60

                if time_diff < gap_time:
                    spikes.loc[this_spike, 'events'] = event_nums[i - 1]
                    event_nums[i] = event_nums[i - 1]

        # Update event_nums after combining events
        event_nums = np.unique(spikes['events'].dropna())

        if len(event_nums) > 0:
            for i in event_nums:
                event_mask = spikes['events'] == i
                first_ob = event_mask.idxmax()
                last_ob = event_mask[::-1].idxmax()

                # Fill in gaps
                spikes.loc[first_ob:last_ob, 'events'] = i

                # Estimate background
                b_left = trimmed_obs.loc[first_ob]
                b_right = trimmed_obs.loc[last_ob]
                b = (b_left + b_right) / 2

                # Remove background from this spike using boolean indexing
                spike_mask = (trimmed_obs.index >= first_ob) & (trimmed_obs.index <= last_ob)
                trimmed_obs[spike_mask] -= b

        # Remove background from all non-spike data
        non_spike_mask = spikes['events'].isna()
        trimmed_obs[non_spike_mask] = 0

        # Set any negative values to zero
        trimmed_obs[trimmed_obs < 0] = 0

        # Save background removed data
        obs.loc[to_keep, j] = trimmed_obs

    return obs


def detect_events(obs, times, gap_time=5, length_threshold=2):
    # Compute maximum concentration across sensors
    max_obs = obs.max(axis=1)

    # Create initial spikes DataFrame with the same index as times
    spikes = pd.DataFrame({'time': times, 'events': max_obs > 0}, index=times)

    # Combine events that are separated by less than gap_time
    to_replace = []
    first_gap = True
    false_seq = []

    # Use times directly since it's a DatetimeIndex
    for i in range(len(times)):
        if not spikes['events'].iloc[i]:
            false_seq.append(times[i])  # Use direct indexing for DatetimeIndex
        else:
            if len(false_seq) <= gap_time and not first_gap:
                to_replace.extend(false_seq)
            first_gap = False
            false_seq = []

    # Use the actual timestamps for indexing
    if to_replace:
        spikes.loc[to_replace, 'events'] = True

    # Replace boolean values with integers to distinguish between events
    spikes['events'] = spikes['events'].map({False: np.nan, True: 0})

    # Assign unique event numbers
    event_num = 0
    prev_value = np.nan

    for idx in spikes.index:
        current_value = spikes.loc[idx, 'events']
        if pd.notna(current_value):
            if pd.isna(prev_value):  # Start of new event
                event_num += 1
            spikes.loc[idx, 'events'] = event_num
        prev_value = current_value

    # Filter events by length threshold
    event_lengths = spikes['events'].value_counts()
    short_events = event_lengths[event_lengths < length_threshold].index
    spikes.loc[spikes['events'].isin(short_events), 'events'] = np.nan

    return spikes, max_obs


def create_visualizations(to_save, event_details, output_path):
    """Create and save various visualizations of the analysis results"""

    # Fix for deprecation warning - helper function
    def safe_float(series):
        """Safely convert single-element series to float"""
        if isinstance(series, pd.Series):
            return float(series.iloc[0])
        return float(series)

    # 1. Event Timeline Plot
    plt.figure(figsize=(15, 6))
    plt.plot(to_save['event_mask'].index, to_save['max_obs'], 'b-', label='Maximum Concentration', alpha=0.6)

    unique_events = event_details['event_number'].unique()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_events)))

    for event, color in zip(unique_events, colors):
        event_mask = to_save['event_mask']['events'] == event
        plt.fill_between(to_save['event_mask'].index[event_mask],
                         to_save['max_obs'][event_mask],
                         color=color, alpha=0.3, label=f'Event {event}')

    plt.title('Detected Methane Events Timeline')
    plt.xlabel('Time')
    plt.ylabel('Concentration (ppm)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'event_timeline.png'))
    plt.close()

    # 2. Enhanced Wind Rose Plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='polar')

    wd_rad = np.radians(to_save['WD'])
    bins = np.linspace(0, 2 * np.pi, 36)
    speeds = to_save['WS']

    # Create speed bins
    speed_bins = [0, 2, 4, 6, 8, np.inf]
    speed_labels = ['0-2', '2-4', '4-6', '6-8', '>8']
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(speed_bins) - 1))

    # Plot each speed bin
    width = 0.8 * 2 * np.pi / len(bins)
    for i in range(len(speed_bins) - 1):
        mask = (speeds >= speed_bins[i]) & (speeds < speed_bins[i + 1])
        hist, _ = np.histogram(wd_rad[mask], bins=bins)
        hist = hist / hist.max() if hist.max() > 0 else hist
        bars = ax.bar(bins[:-1], hist, width=width, bottom=0.0,
                      color=colors[i], alpha=0.7, label=f'{speed_labels[i]} m/s')

    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N')
    plt.legend(title='Wind Speed (m/s)', bbox_to_anchor=(1.2, 0.95))
    ax.set_title('Enhanced Wind Rose')
    plt.savefig(os.path.join(output_path, 'wind_rose_enhanced.png'), bbox_inches='tight')
    plt.close()

    # 3. Emission Rate Plot (with fixed warning)
    plt.figure(figsize=(12, 6))
    plt.errorbar(range(len(event_details)),
                 event_details['emission_rate'].apply(safe_float),
                 yerr=[
                     event_details['emission_rate'].apply(safe_float) - event_details['error_lower'].apply(safe_float),
                     event_details['error_upper'].apply(safe_float) - event_details['emission_rate'].apply(safe_float)],
                 fmt='o', capsize=5, capthick=1, elinewidth=1, markersize=8)

    plt.grid(True, alpha=0.3)
    plt.title('Emission Rate Estimates with Uncertainty')
    plt.xlabel('Event Number')
    plt.ylabel('Emission Rate (kg/hr)')

    for i, source in enumerate(event_details['source_location']):
        plt.annotate(source, (i, safe_float(event_details['emission_rate'].iloc[i])),
                     xytext=(0, 10), textcoords='offset points',
                     ha='center', va='bottom', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'emission_rates.png'))
    plt.close()

    # 4. Concentration Heatmap Over Time
    plt.figure(figsize=(15, 8))
    event_times = to_save['event_mask'].index

    # Create a heatmap of all sensor readings
    sensor_data = pd.DataFrame(index=event_times)
    for source in to_save['source_names']:
        sensor_data[source] = to_save['max_obs']

    plt.imshow(sensor_data.T, aspect='auto', cmap='YlOrRd')
    plt.colorbar(label='Concentration (ppm)')
    plt.title('Sensor Readings Heatmap')
    plt.xlabel('Time Index')
    plt.ylabel('Source Location')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'concentration_heatmap.png'))
    plt.close()

    # 5. Wind Speed vs Concentration Scatter Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(to_save['WS'], to_save['max_obs'], alpha=0.5)
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('Concentration (ppm)')
    plt.title('Wind Speed vs Concentration')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'wind_speed_concentration.png'))
    plt.close()

    # 6.Event Duration Analysis
    plt.figure(figsize=(12, 6))
    event_durations = []
    for event in unique_events:
        event_mask = to_save['event_mask']['events'] == event
        duration = (event_times[event_mask][-1] - event_times[event_mask][0]).total_seconds() / 60  # in minutes
        event_durations.append(duration)

    plt.bar(range(len(event_durations)), event_durations)
    plt.title('Event Durations')
    plt.xlabel('Event Number')
    plt.ylabel('Duration (minutes)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'event_durations.png'))
    plt.close()

def main():
    # Configuration
    config = {
        'gap_time': 5,
        'length_threshold': 2,
        'simulation_data_path': "./input_data_step2/simulation_output_new.pkl",
        'output_file_path': "path to save output",
    }

    # Create output directory
    os.makedirs(config['output_file_path'], exist_ok=True)

    # Read simulation data
    logging.info("Reading simulation data...")
    try:
        with open(config['simulation_data_path'], 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        logging.error(f"Error reading simulation data: {str(e)}")
        logging.error(traceback.format_exc())
        return

    obs = pd.DataFrame(data['obs'])
    times = pd.to_datetime(data['times'])
    sims = {k: pd.DataFrame(v) for k, v in data.items() if k not in ['obs', 'times', 'WD', 'WS']}

    # Step 1: Save raw input data
    obs.to_csv(os.path.join(config['output_file_path'], "step1_raw_observations.csv"))
    pd.DataFrame({'times': times}).to_csv(os.path.join(config['output_file_path'], "step1_timestamps_new.csv"))

    obs = remove_background(obs, times, gap_time=config['gap_time'], amp_threshold=0.75)
    obs.to_csv(os.path.join(config['output_file_path'], "step2_background_removed_obs_new.csv"))

    # STEP 3: Event detection
    logging.info("Detecting events...")
    spikes, max_obs = detect_events(obs, times, gap_time=config['gap_time'],
                                    length_threshold=config['length_threshold'])

    # Get unique event numbers (excluding NaN)
    event_nums = pd.Series(spikes['events'].unique())
    event_nums = event_nums.dropna().astype(int).sort_values().tolist()

    # Format the results to match R output exactly
    event_results = pd.DataFrame({
        'time': times.strftime('%Y-%m-%d %H:%M:%S'),
        'max_concentration': max_obs.round(4),
        'event_number': spikes['events'].astype('Int64')
    })

    # Replace NaN with 'NA' string to match R output
    event_results['event_number'] = event_results['event_number'].astype(str).replace('nan', 'NA')

    event_results.to_csv(os.path.join(config['output_file_path'], "step3_event_detection_new.csv"),
                         index=False)

    # STEP 4: Compute alignment metric
    logging.info("Computing alignment metrics...")
    metrics = pd.DataFrame(index=range(len(event_nums)), columns=sims.keys())


    # STEP 4: Compute alignment metric
    logging.info("Computing alignment metrics...")
    metrics = pd.DataFrame(index=range(len(event_nums)), columns=sims.keys())

    # Ensure that all DataFrames have the same time index
    common_index = times
    obs = obs.set_index(common_index)
    for s in sims.keys():
        sims[s] = sims[s].set_index(common_index)

    for t, event_num in enumerate(event_nums):
        event_mask = spikes['events'] == event_num
        event_times = times[event_mask]
        if len(event_times) > 0:
            start_time = event_times.min()
            end_time = event_times.max()
            this_mask = (times >= start_time) & (times <= end_time)

            for s in sims.keys():
                preds = sims[s]
                all_obs_to_compare = []
                all_preds_to_compare = []

                for r in obs.columns:
                    obs_int = obs.loc[this_mask, r]
                    preds_int = preds.loc[this_mask, r]

                    to_compare = ~obs_int.isna()

                    all_obs_to_compare.extend(obs_int[to_compare])
                    all_preds_to_compare.extend(preds_int[to_compare])

                if len(all_obs_to_compare) > 0:
                    x = np.array(all_preds_to_compare)
                    y = np.array(all_obs_to_compare)

                    x[x < 1e-30] = 0
                    y[y < 1e-30] = 0

                    if not all(x == 0) and not all(y == 0):
                        metrics.loc[t, s] = np.corrcoef(x, y)[0, 1]

    # Use infer_objects instead of fillna
    metrics = metrics.infer_objects()
    metrics['event_number'] = event_nums
    metrics.to_csv(os.path.join(config['output_file_path'], "step4_alignment_metrics_py.csv"))

    logging.debug(f"Metrics shape: {metrics.shape}")
    logging.debug(f"Metrics columns: {metrics.columns}")
    logging.debug(f"Metrics head: \n{metrics.head()}")

    # STEP 5: COMPUTE LOCALIZATION AND QUANTIFICATION ESTIMATES
    logging.info("Computing localization and quantification estimates...")

    rate_est_all_events = np.full(len(event_nums), np.nan)
    loc_est_all_events = np.full(len(event_nums), '', dtype=object)
    error_lower_all_events = np.full(len(event_nums), np.nan)
    error_upper_all_events = np.full(len(event_nums), np.nan)

    all_preds_to_compare_all_events = []
    all_obs_to_compare_all_events = []

    for t, event_num in enumerate(event_nums):
        logging.info(f"Processing event {t + 1}/{len(event_nums)}")

        these_metrics = metrics.iloc[t].drop('event_number')

        logging.debug(f"Metrics for event {t + 1}: {these_metrics}")

        # Check if these_metrics is empty or contains only NaN values
        if these_metrics.empty or these_metrics.isna().all():
            logging.warning(f"No valid metrics for event {t + 1}. Skipping this event.")
            continue

        # Find the index of the maximum non-NaN value
        max_index = these_metrics.idxmax()

        if pd.isna(max_index):
            logging.warning(f"No valid maximum metric found for event {t + 1}. Skipping this event.")
            continue

        loc_est_all_events[t] = max_index

        event_mask = spikes['events'] == event_num
        event_times = times[event_mask]

        if len(event_times) == 0:
            logging.warning(f"No event times found for event {t + 1}. Skipping this event.")
            continue

        start_time = event_times.min()
        end_time = event_times.max()
        this_mask = (times >= start_time) & (times <= end_time)

        if loc_est_all_events[t] not in sims:
            logging.warning(f"Invalid simulation key {loc_est_all_events[t]} for event {t + 1}. Skipping this event.")
            continue

        event_preds = sims[loc_est_all_events[t]].loc[this_mask]
        event_obs = obs.loc[this_mask]

        all_preds_to_compare = []
        all_obs_to_compare = []

        for r in obs.columns:
            these_preds = event_preds[r]
            these_obs = event_obs[r]

            if these_obs.isna().all():
                continue

            # Ensure event_times index matches these_preds and these_obs
            aligned_event_times = event_times[event_times.isin(these_preds.index)]

            logging.debug(
                f"Column {r} - Preds shape: {these_preds.shape}, Obs shape: {these_obs.shape}, Event times shape: {aligned_event_times.shape}")
            logging.debug(f"Column {r} - Preds index: {these_preds.index}")
            logging.debug(f"Column {r} - Obs index: {these_obs.index}")
            logging.debug(f"Column {r} - Event times: {aligned_event_times}")

            try:
                preds_spikes = find_spikes(aligned_event_times, these_preds.loc[aligned_event_times], amp_threshold=1,
                                           make_plot=False)
                obs_spikes = find_spikes(aligned_event_times, these_obs.loc[aligned_event_times], amp_threshold=1,
                                         make_plot=False)

                logging.debug(
                    f"Column {r} - Preds spikes shape: {preds_spikes.shape}, Obs spikes shape: {obs_spikes.shape}")

                both_in_spike_mask = ~preds_spikes['events'].isna() & ~obs_spikes['events'].isna()

                logging.debug(f"Column {r} - Both in spike mask sum: {both_in_spike_mask.sum()}")

                if both_in_spike_mask.any():
                    # Reset index to avoid KeyError
                    these_preds = these_preds.reset_index(drop=True)
                    these_obs = these_obs.reset_index(drop=True)
                    both_in_spike_mask = both_in_spike_mask.reset_index(drop=True)

                    all_preds_to_compare.extend(these_preds[both_in_spike_mask])
                    all_obs_to_compare.extend(these_obs[both_in_spike_mask])

            except Exception as e:
                logging.error(f"Error processing column {r} for event {t + 1}: {str(e)}")
                logging.error(traceback.format_exc())
                continue

        logging.debug(
            f"Event {t + 1} - All preds to compare length: {len(all_preds_to_compare)}, All obs to compare length: {len(all_obs_to_compare)}")

        if len(all_preds_to_compare) > 4:
            all_preds_to_compare_all_events.extend(all_preds_to_compare)
            all_obs_to_compare_all_events.extend(all_obs_to_compare)

            n_samples = 1000
            q_vals = []

            for _ in range(n_samples):
                this_sample = np.random.choice(len(all_preds_to_compare), size=len(all_preds_to_compare) // 2,
                                               replace=True)

                q_grid = np.logspace(np.log10(0.0001), np.log10(3000), num=2000)
                sse = []

                for q in q_grid:
                    qxp = q * np.array(all_preds_to_compare)
                    sse.append(np.sqrt(np.mean((np.array(all_obs_to_compare)[this_sample] - qxp[this_sample]) ** 2)))

                if np.all(np.diff(sse) > 0):
                    q_vals.append(np.nan)
                else:
                    q_vals.append(q_grid[np.argmin(sse)])

            rate_est_all_events[t] = np.nanmean(q_vals) * 3600
            error_lower_all_events[t] = np.nanquantile(q_vals, 0.05) * 3600
            error_upper_all_events[t] = np.nanquantile(q_vals, 0.95) * 3600

    # Check that predictions and observations are on the same order of magnitude
    med_p = np.median(all_preds_to_compare_all_events)
    med_o = np.median(all_obs_to_compare_all_events)

    if med_p > med_o * 10 or med_p < med_o / 10:
        if med_p > med_o:
            logging.warning(
                "Simulation output and observations are on different order, recommend re-simulating with smaller rate")
        else:
            logging.warning(
                "Simulation output and observations are on different order, recommend re-simulating with larger rate")

    # Package up event detection, localization, and quantification results
    to_save = {
        'event_mask': spikes,
        'max_obs': max_obs,
        'localization_estimates': loc_est_all_events,
        'rate_estimates': rate_est_all_events,
        'error_lower': error_lower_all_events,
        'error_upper': error_upper_all_events,
        'source_names': list(sims.keys()),
        'WD': data['WD'],
        'WS': data['WS']
    }

    # Save detailed results after each event
    event_details = pd.DataFrame({
        'event_number': event_nums,
        'source_location': loc_est_all_events,
        'emission_rate': rate_est_all_events,
        'error_lower': error_lower_all_events,
        'error_upper': error_upper_all_events
    })
    event_details.to_csv(os.path.join(config['output_file_path'], "step5_event_details_new.csv"), index=False)

    # Save results
    with open(os.path.join(config['output_file_path'], "step5_results_new.pkl"), 'wb') as f:
        pickle.dump(to_save, f)

    # Create visualizations
    create_visualizations(to_save, event_details, config['output_file_path'])

    logging.info("Analysis completed successfully.")


if __name__ == "__main__":
    main()























































