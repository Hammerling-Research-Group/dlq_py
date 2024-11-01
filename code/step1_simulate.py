import os
import pandas as pd
import multiprocessing as mp
from scipy.interpolate import interp1d
from concurrent.futures import ProcessPoolExecutor
import pickle
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from datetime import datetime
from functools import partial
from math import sqrt, pi, cos, sin, atan2
import pytz
from typing import List, Tuple

# Define helper functions (you'll need to implement these separately)
from helper_distance_conversions import latlon_to_utm
from helper_gpuff_function import gpuff, get_stab_class

# Set Pandas and NumPy options for consistent display of small numbers
np.set_printoptions(precision=6, suppress=True)
pd.set_option('display.float_format', '{:.6e}'.format)

# START USER INPUT
#---------------------------------------------------------------------------

# Set path to your CSV file directly
raw_sensor_observations_path = "./input_data_step1/ADED_data_clean.csv"

# Set output directory for CSV files
output_directory = 'path to output_data folder/'

# END OF USER INPUT - NO MODIFICATION NECESSARY BELOW THIS POINT
#---------------------------------------------------------------------------

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# STEP 1: SET UP PARAMETERS AND DIRECTORY STRUCTURE
#---------------------------------------------------------------------------

# Set parameters as per your input
num_cores_to_use = 2
dt = 1  # Time step for simulation in seconds
cutoff_t = 20  # Cutoff time in seconds
ignore_dist = 80  # Ignore distance in meters
chunk_size = 54  # Chunk size for processing data
emission_rate = 0.001  # Emission rate in units per second
run_mode = "constant"
# Set timezone
# Ensure consistent timezone usage
tz = pytz.timezone("America/Denver")
# Apply timezone consistently when parsing dates

# Set simulation start and end times
start_time = pd.Timestamp("2022-05-12 20:00:00", tz=tz)
end_time = pd.Timestamp("2022-05-12 22:00:00", tz=tz)


# Start code timer
code_start_time = datetime.now()

# STEP 2: READ IN CMS SENSOR DATA
#---------------------------------------------------------------------------

# Read in sensor data csv
raw_data = pd.read_csv(raw_sensor_observations_path, parse_dates=['time'])

# Convert 'time' column to datetime and localize to the specified timezone
raw_data['time'] = pd.to_datetime(raw_data['time'], utc=True).dt.tz_convert(tz)

# Handle DST transitions
raw_data['time'] = raw_data['time'].dt.tz_localize(None).dt.tz_localize(tz, ambiguous='infer', nonexistent='shift_forward')

# Mask data within start and end times
raw_data = raw_data[(raw_data['time'] >= start_time) & (raw_data['time'] <= end_time)]

# Extract date for alignment with R output
raw_data['date'] = raw_data['time'].dt.date
# STEP 3: READ IN SITE GEOMETRY AND SETUP DATA STRUCTURE TO EXPORT
#---------------------------------------------------------------------------

# Read in sensor location csv (replace this with your actual file path)
sensor_locations_path = "./input_data_step1/sensor_locations.csv"
sensor_locs = pd.read_csv(sensor_locations_path)
sensor_locs = sensor_locs.sort_values('name')
n_r = len(sensor_locs)

# Read in source location csv (replace this with your actual file path)
source_locations_path = "./input_data_step1/source_locations.csv"
source_locs = pd.read_csv(source_locations_path)
n_s = len(source_locs)

# Initialize data structure to store simulation output
data_to_save = {
    "times": None,
    "obs": None,
    "WD": None,
    "WS": None,
    **{name: None for name in source_locs['name']}
}
# STEP 4: COMBINE WIND DATA ACROSS SENSORS THAT MEASURE WIND
# Figure out which units collect wind data
n_r = len(sensor_locs)
wind_unit = np.array(
    [raw_data[raw_data['name'] == sensor_locs['name'].iloc[r]]['wind.speed'].notna().any() for r in range(n_r)])

# Create a sequence of minutes from start time to end time
min_seq = pd.date_range(start=raw_data['time'].min(), end=raw_data['time'].max(), freq='min')


# Function to process each time chunk
def process_time_chunk(t):
    these_wd = np.full(sum(wind_unit), np.nan)
    these_ws = np.full(sum(wind_unit), np.nan)
    ind = 0
    for r in np.where(wind_unit)[0]:
        this_mask = (raw_data['name'] == sensor_locs['name'].iloc[r]) & (raw_data['time'] == min_seq[t])

        if this_mask.sum() > 0:
            these_wd[ind] = raw_data.loc[this_mask, 'wind.direction'].values[0]
            these_ws[ind] = raw_data.loc[this_mask, 'wind.speed'].values[0]
        ind += 1
    return these_ws, these_wd


# Fire up the parallel processing
with ProcessPoolExecutor() as executor:
    wind_list = list(executor.map(process_time_chunk, range(len(min_seq))))

# Parse output of the map function
out = np.array(wind_list)
WS = np.vstack(out[:, 0])
WD = np.vstack(out[:, 1])

# Gap fill the WS (using interpolation)
WS = pd.DataFrame(WS).interpolate(method='linear', limit=5).to_numpy()

# Take median of WS across sensors
WS = np.nanmedian(WS, axis=1)

# Convert WD to radians and compute horizontal and vertical components
WD = WD * np.pi / 180
WD_x = np.cos(WD)
WD_y = np.sin(WD)

# Gap fill x and y components separately
WD_x = pd.DataFrame(WD_x).interpolate(method='linear', limit=5).to_numpy()
WD_y = pd.DataFrame(WD_y).interpolate(method='linear', limit=5).to_numpy()

# Take median across sensors of x and y components separately
WD_x = np.nanmedian(WD_x, axis=1)
WD_y = np.nanmedian(WD_y, axis=1)

# Recombine into an angle and clean up
WD = np.arctan2(WD_y, WD_x) * 180 / np.pi
WD = np.where(WD < 0, WD + 360, WD)

# Convert wind direction from clockwise with 0 deg at north to
# counterclockwise with 0 deg at east (traditional angle definition)
WD = 90 - WD
WD = np.where(WD < 0, WD + 360, WD)

# Convert wind direction from direction wind is coming from to
# direction wind is going
WD = 180 + WD
WD = np.where(WD >= 360, WD - 360, WD)

# Convert wind direction to radians
WD = WD * np.pi / 180
# Save WS and WD
wind_data = pd.DataFrame({'times': min_seq, 'WS': WS, 'WD': WD})
wind_data.to_csv("wind_data1.csv", index=False)
wind_data.to_pickle("wind_data1.pkl")
# STEP 5: MOVE DATA INTO TIME-ALIGNED TABLE WITH SENSORS AS COLUMNS
#---------------------------------------------------------------------------

# Set up data table to hold cleaned data
data = pd.DataFrame(index=min_seq, columns=list(sensor_locs['name']) + ['WS', 'WD'])
data['WS'] = WS
data['WD'] = WD

def process_row(t):
    row = []
    for r in range(len(sensor_locs)):
        mask = (raw_data['name'] == sensor_locs['name'].iloc[r]) & (raw_data['time'] == min_seq[t])
        if mask.any():
            row.append(raw_data.loc[mask, 'methane'].iloc[0])
        else:
            row.append(np.nan)
    return row

# Use multiprocessing to process rows
with mp.Pool(num_cores_to_use) as pool:
    row_list = pool.map(process_row, range(len(min_seq)))

# Combine output of parallel processing into one dataframe
data.iloc[:, :len(sensor_locs)] = row_list

# Remove observations that still have no wind data
data = data.dropna(subset=['WS'])

# Pull out times
times = data.index

# Pull out just the methane observations
obs = data.iloc[:, :len(sensor_locs)]

# Clean up
del raw_data, wind_unit, WD_x, WD_y

# Optional: Save results to CSV or another format if needed
data.to_csv('path_to_save_results.csv', index=True)
data.to_csv(os.path.join(output_directory, 'aligned_data.csv'))

# STEP 6: LOOP THROUGH POTENTIAL SOURCES AND SIMULATE CONCENTRATION VALUES AT RECEPTORS
# ---------------------------------------------------------------------------

def est_on_fine_grid(y: np.ndarray, dt: int) -> np.ndarray:
    """
    Improved interpolation function to match R behavior
    """
    # Ensure we're working with float64 for maximum precision
    y = y.astype(np.float64)

    out = []
    for i in range(len(y) - 1):
        # Create exactly 60/dt points between each pair of values
        t = np.linspace(0, 1, int(60 / dt), endpoint=False)
        interp_y = y[i] + (y[i + 1] - y[i]) * t
        out.extend(interp_y)

    # Add the last value
    out.append(y[-1])

    return np.array(out)


def process_chunk(h: int, n_ints: int, chunk_size: int, dt: int, cutoff_t: int,
                  source_x: float, source_y: float, source_z: float,
                  x_r: np.ndarray, y_r: np.ndarray, z_r: np.ndarray, n_r: int,
                  Q_truth: np.ndarray, WS_x: np.ndarray, WS_y: np.ndarray,
                  WS: np.ndarray, high_res_times, ignore_dist: float,
                  run_mode: str, n_chunks: int) -> Tuple[np.ndarray, np.ndarray]:
    # Use float64 for all calculations
    x_r = x_r.astype(np.float64)
    y_r = y_r.astype(np.float64)
    z_r = z_r.astype(np.float64)
    WS_x = WS_x.astype(np.float64)
    WS_y = WS_y.astype(np.float64)
    WS = WS.astype(np.float64)

    # Standard chunk size at simulation frequency
    standard_chunk_size = chunk_size * 60 // dt

    # First index of this chunk at simulation frequency
    this_chunk_start = 1 + (h - 1) * standard_chunk_size

    # Last index of this chunk at simulation frequency
    this_chunk_end = min(h * standard_chunk_size, n_ints)

    # Last index of the extended chunk to capture stragglers
    ext_chunk_end = min(h * standard_chunk_size + cutoff_t * 60 // dt - 1, n_ints)

    # Calculate actual lengths
    this_chunk_length = this_chunk_end - (this_chunk_start - 1)
    ext_chunk_length = ext_chunk_end - (this_chunk_start - 1)

    # Create slice objects
    this_chunk_mask = slice(this_chunk_start - 1, this_chunk_end)
    ext_chunk_mask = slice(this_chunk_start - 1, ext_chunk_end)

    # Initialize matrices to hold puff locations and distance traveled
    puff_x_locs = np.full((cutoff_t * (60 // dt), ext_chunk_length), np.nan)
    puff_y_locs = np.full((cutoff_t * (60 // dt), ext_chunk_length), np.nan)
    total_dist = np.full((cutoff_t * (60 // dt), ext_chunk_length), np.nan)

    # Compute first location and distance based on source location
    puff_x_locs[0, :] = source_x
    puff_y_locs[0, :] = source_y
    total_dist[0, :] = 0

    # Fill in remainder of location and distance matrices using previous locations
    for p in range(puff_x_locs.shape[1]):  # Loop through puffs
        for t in range(1, puff_x_locs.shape[0]):  # Loop through time steps that this puff is alive
            # Compute movement
            if run_mode == "dynamic":
                x_movement = WS_x[ext_chunk_mask][t + p - 1] * dt
                y_movement = WS_y[ext_chunk_mask][t + p - 1] * dt
            elif run_mode == "constant":
                x_movement = WS_x[ext_chunk_mask][p] * dt
                y_movement = WS_y[ext_chunk_mask][p] * dt
            else:
                print("Invalid run mode")
                return None, None

            # Compute new location and distance
            puff_x_locs[t, p] = puff_x_locs[t - 1, p] + x_movement
            puff_y_locs[t, p] = puff_y_locs[t - 1, p] + y_movement
            total_dist[t, p] = total_dist[t - 1, p] + sqrt(x_movement ** 2 + y_movement ** 2)

    if run_mode == "constant":
        # Initialize list to hold stability classes at each time step.
        stab_classes = []

        # Loop through times, get time and WS, get stab class, save stab class
        for j in range(ext_chunk_end - (this_chunk_start - 1)):
            # Get time
            time_to_use = high_res_times[ext_chunk_mask.start + j]

            # Get wind speed
            WS_to_use = WS[ext_chunk_mask][j]

            # Get stability class based on time and wind speed
            stab_classes.append(get_stab_class(WS_to_use, time_to_use))

    # Initialize matrix to hold concentration values for each receptor for each time step
    C = np.zeros((min(this_chunk_length, len(Q_truth[this_chunk_mask])), n_r))

    # Initialize variable to hold previous time used to get stability class
    previous_time = None

    # Iterate through time steps at simulation frequency
    for j in range(min(ext_chunk_end - (this_chunk_start - 1), len(C))):
        if run_mode == "dynamic":
            # Get time to use for stability class
            time_to_use = high_res_times[this_chunk_start - 1 + j]

            # If time is NA (happens during daylight savings time transition) use previous time
            time_to_use = previous_time if pd.isna(time_to_use) else time_to_use

            # Get the stability class based on wind speed and time of day
            stab_class = get_stab_class(U=WS[this_chunk_mask][j], time=time_to_use)

            # Reset previous time
            previous_time = time_to_use

        # Vector to hold concentration predictions at the jth time step
        this_C = np.zeros(n_r)

        # Loop through puffs in existence at the jth time step
        for k in range(1, cutoff_t * 60 // dt + 1):
            # Compute matrix indices for the kth puff in the pre-computed location matrices
            col_it = j - k + 1
            row_it = k - 1

            # End loop over puffs after all puffs are accounted for.
            if col_it < 0:
                break

            # Get location of the kth puff and the total distance it has traveled
            this_puff_x = puff_x_locs[row_it, col_it]
            this_puff_y = puff_y_locs[row_it, col_it]
            this_total_dist = total_dist[row_it, col_it]

            if run_mode == "constant":
                stab_class = stab_classes[col_it]

            # Compute distance between this puff and all receptors
            distances = np.sqrt((x_r - this_puff_x) ** 2 + (y_r - this_puff_y) ** 2)

            # Skip this puff if smallest distance is greater than ignore_dist
            if np.min(distances) > ignore_dist:
                continue

            # Compute the concentration at each receptor location from the kth puff
            if j < len(Q_truth[this_chunk_mask]):
                this_C += dt * gpuff(Q=Q_truth[this_chunk_mask][j],
                                     stab_class=stab_class,
                                     x_p=this_puff_x,
                                     y_p=this_puff_y,
                                     x_r_vec=x_r,
                                     y_r_vec=y_r,
                                     z_r_vec=z_r,
                                     total_dist=this_total_dist,
                                     H=source_z,
                                     U=WS[this_chunk_mask][j])

        if j < len(C):
            C[j, :] = this_C

    C_to_pass = None
    if h != n_chunks:
        # Initialize matrix to hold concentration values for each receptor for each time step
        C_to_pass = np.zeros((ext_chunk_end - this_chunk_end, n_r))

        # Initialize variable to hold previous time used to get stability class
        previous_time = None

        this_it = 0

        # Iterate through extra time steps at simulation frequency to get bottom right corner
        for j in range(this_chunk_length, ext_chunk_length):
            if run_mode == "dynamic":
                # Get time to use for stability class
                time_to_use = high_res_times[ext_chunk_mask.start + j]

                # If time is NA (happens during daylight savings time transition) use previous time
                time_to_use = previous_time if pd.isna(time_to_use) else time_to_use

                # Get the stability class based on wind speed and time of day
                stab_class = get_stab_class(U=WS[ext_chunk_mask][j], time=time_to_use)

                # Reset previous time
                previous_time = time_to_use

            # Vector to hold concentration predictions at the jth time step
            this_C = np.zeros(n_r)

            # Loop through puffs in existence at the jth time step
            for k in range(1, cutoff_t * 60 // dt + 1):
                # Compute matrix indices for the kth puff in the pre-computed location matrices
                col_it = j - k + 1
                row_it = k - 1

                if col_it >= this_chunk_length:
                    continue

                # End loop over puffs after all puffs are accounted for.
                if col_it < 0:
                    break

                # Get location of the kth puff and the total distance it has traveled
                this_puff_x = puff_x_locs[row_it, col_it]
                this_puff_y = puff_y_locs[row_it, col_it]
                this_total_dist = total_dist[row_it, col_it]

                if run_mode == "constant":
                    stab_class = stab_classes[col_it]

                # Compute distance between this puff and all receptors
                distances = np.sqrt((x_r - this_puff_x) ** 2 + (y_r - this_puff_y) ** 2)

                # Skip this puff if smallest distance is greater than ignore_dist
                if np.min(distances) > ignore_dist:
                    continue

                # Compute the concentration at each receptor location from the kth puff
                this_C += dt * gpuff(Q=Q_truth[ext_chunk_mask][j],
                                     stab_class=stab_class,
                                     x_p=this_puff_x,
                                     y_p=this_puff_y,
                                     x_r_vec=x_r,
                                     y_r_vec=y_r,
                                     z_r_vec=z_r,
                                     total_dist=this_total_dist,
                                     H=source_z,
                                     U=WS[ext_chunk_mask][j])

            C_to_pass[this_it, :] = this_C
            this_it += 1

    return C, C_to_pass


def run_source_simulations(data: pd.DataFrame, source_locs: pd.DataFrame, sensor_locs: pd.DataFrame,
                           times: pd.DatetimeIndex, emission_rate: float, dt: int, output_directory: str,
                           num_cores_to_use: int, cutoff_t: int, ignore_dist: float, chunk_size: int,
                           run_mode: str, tz: str):
    # Create main simulation outputs directory
    os.makedirs(os.path.join(output_directory, "simulation_outputs_new"), exist_ok=True)

    # Initialize dictionary to store all simulation data
    data_to_save = {
        "times": times,
        "obs": data.iloc[:, :len(sensor_locs)],  # Assuming first columns are observations
        "WD": data['WD'].values,
        "WS": data['WS'].values
    }

    for s in range(len(source_locs)):
        print(f"Processing source: {s + 1}/{len(source_locs)}")

        # Create source-specific directory
        source_dir = os.path.join(output_directory, "simulation_outputs_new", f"source_{s + 1}")
        os.makedirs(source_dir, exist_ok=True)

        # 1. Source location processing
        source_lon = source_locs['lon'].iloc[s]
        source_lat = source_locs['lat'].iloc[s]
        source_x, source_y, _, _ = latlon_to_utm(source_lat, source_lon)
        source_z = source_locs['height'].iloc[s]

        source_location_df = pd.DataFrame({
            'longitude': [source_lon],
            'latitude': [source_lat],
            'utm_x': [source_x],
            'utm_y': [source_y],
            'height': [source_z]
        })
        source_location_df.to_csv(os.path.join(source_dir, "1_source_location.csv"), index=False)

        # 2. Receptor locations processing
        x_r = []
        y_r = []
        z_r = sensor_locs['height'].values
        for _, row in sensor_locs.iterrows():
            utm_x, utm_y, _, _ = latlon_to_utm(row['lat'], row['lon'])
            x_r.append(utm_x)
            y_r.append(utm_y)

        x_r = np.array(x_r)
        y_r = np.array(y_r)
        n_r = len(sensor_locs)

        receptor_data = pd.DataFrame({
            'name': sensor_locs['name'],
            'longitude': sensor_locs['lon'],
            'latitude': sensor_locs['lat'],
            'height': z_r,
            'utm_x': x_r,
            'utm_y': y_r
        })
        receptor_data.to_csv(os.path.join(source_dir, "2_receptor_locations.csv"), index=False)

        # 3-5. Process high resolution data
        WA = data['WD'].values
        WS = data['WS'].values
        Q_truth = np.full(len(times), emission_rate)

        # Get x and y components of wind angle
        WA_x = np.cos(WA)
        WA_y = np.sin(WA)

        # Interpolate to simulation frequency
        WA_x_high_res = est_on_fine_grid(WA_x, dt)
        WA_y_high_res = est_on_fine_grid(WA_y, dt)

        # Bring back to angle
        WA_high_res = np.arctan2(WA_y_high_res, WA_x_high_res)
        WA_high_res = np.where(WA_high_res < 0, WA_high_res + (2 * np.pi), WA_high_res)
        WA_high_res = np.where(WA_high_res >= 2 * np.pi, WA_high_res - (2 * np.pi), WA_high_res)

        # Increase frequency of WS and Q
        WS_high_res = est_on_fine_grid(WS, dt)
        Q_truth_high_res = est_on_fine_grid(Q_truth, dt)

        # Get x and y components of wind speed
        WS_x = np.cos(WA_high_res) * WS_high_res
        WS_y = np.sin(WA_high_res) * WS_high_res

        # Create high resolution timestamps
        n_ints = len(WA_high_res)
        high_res_times = pd.date_range(start=times[0], end=times[-1], periods=n_ints)

        # Save high resolution data
        high_res_df = pd.DataFrame({
            'time': high_res_times,
            'wind_angle': WA_high_res,
            'wind_speed': WS_high_res,
            'wind_speed_x': WS_x,
            'wind_speed_y': WS_y,
            'emission_rate': Q_truth_high_res
        })
        high_res_df.to_csv(os.path.join(source_dir, "5_high_resolution_data.csv"), index=False)

        # Compute number of time chunks
        n_chunks = int(np.ceil(n_ints / (chunk_size * (60 / dt))))
        # Prepare arguments for process_chunk
        process_chunk_partial = partial(
            process_chunk,
            n_ints=n_ints,
            chunk_size=chunk_size,
            dt=dt,
            cutoff_t=cutoff_t,
            source_x=source_x,
            source_y=source_y,
            source_z=source_z,
            x_r=x_r,
            y_r=y_r,
            z_r=z_r,
            n_r=n_r,
            Q_truth=Q_truth_high_res,
            WS_x=WS_x,
            WS_y=WS_y,
            WS=WS_high_res,
            high_res_times=high_res_times,
            ignore_dist=ignore_dist,
            run_mode=run_mode,
            n_chunks=n_chunks
        )

        # Fire up the parallel cluster
        with ProcessPoolExecutor(max_workers=num_cores_to_use) as executor:
            big_C_list = list(executor.map(process_chunk_partial, range(1, n_chunks + 1)))

        # Add the bottom right corner to the next time chunk
        if n_chunks > 1:
            for h in range(1, n_chunks):
                correction = big_C_list[h - 1][1]
                if correction is not None:
                    original_C = big_C_list[h][0][:len(correction)]
                    big_C_list[h] = (
                        np.vstack([original_C + correction, big_C_list[h][0][len(correction):]]),
                        big_C_list[h][1]
                    )

        # Extract data from list object
        big_C = np.vstack([chunk[0] for chunk in big_C_list])

        # Average the concentration predictions back up to a one-minute resolution
        C_avg = np.zeros((len(times), n_r))

        # First entry corresponds to last second of the first minute
        C_avg[0, :] = big_C[0, :]

        # Fill in the rest of the matrix
        for j in range(1, len(C_avg)):
            this_mask = slice((j - 1) * (60 // dt), j * (60 // dt))
            C_avg[j, :] = np.mean(big_C[this_mask, :], axis=0)

        # Save final averaged concentrations
        final_conc_df = pd.DataFrame(C_avg, columns=receptor_data['name'])
        final_conc_df['time'] = times
        final_conc_df.to_csv(os.path.join(source_dir, "8_final_concentrations.csv"), index=False)
        tmp = pd.DataFrame(C_avg, columns=sensor_locs['name'])
        data_to_save[source_locs['name'].iloc[s]] = tmp

        # Save the complete simulation data as pickle
    output_pkl_path = os.path.join(output_directory, "simulation_output_new.pkl")
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(data_to_save, f)

    # Calculate and print execution time
    code_stop_time = datetime.now()
    execution_time = code_stop_time - code_start_time
    print(f"Execution time: {execution_time.total_seconds() / 60:.2f} minutes")

    return output_pkl_path


# At the end of your script, modify the main execution block:
if __name__ == "__main__":
    output_file = run_source_simulations(
        data=data,
        source_locs=source_locs,
        sensor_locs=sensor_locs,
        times=times,
        emission_rate=emission_rate,
        dt=dt,
        output_directory=output_directory,
        num_cores_to_use=num_cores_to_use,
        cutoff_t=cutoff_t,
        ignore_dist=ignore_dist,
        chunk_size=chunk_size,
        run_mode=run_mode,
        tz=tz
    )
    print(f"Simulation data saved to: {output_file}")

