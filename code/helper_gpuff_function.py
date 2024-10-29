import numpy as np
from datetime import datetime
from typing import List, Tuple, Union


def is_day(time: datetime) -> bool:
    time_hour = time.hour
    return 7 <= time_hour <= 18


def get_stab_class(U: float, time: datetime) -> List[str]:
    if U < 2:
        return ['A', 'B'] if is_day(time) else ['E', 'F']
    elif 2 <= U < 3:
        return ['B'] if is_day(time) else ['E', 'F']
    elif 3 <= U < 5:
        return ['B', 'C'] if is_day(time) else ['D', 'E']
    elif 5 <= U < 6:
        return ['C', 'D'] if is_day(time) else ['D']
    else:
        return ['D']


def compute_sigma_vals(stab_class: List[str], total_dist: float) -> Tuple[float, float]:
    sigma_y_vals = []
    sigma_z_vals = []

    for sc in stab_class:
        if sc == 'A':
            if total_dist <= 0.1:
                a, b = 122.8, 0.9447
            elif total_dist <= 0.15:
                a, b = 158.08, 1.0542
            elif total_dist <= 0.20:
                a, b = 170.22, 1.0932
            elif total_dist <= 0.25:
                a, b = 179.52, 1.1262
            elif total_dist <= 0.3:
                a, b = 217.41, 1.2644
            elif total_dist <= 0.4:
                a, b = 258.89, 1.4094
            elif total_dist <= 0.5:
                a, b = 346.75, 1.7283
            else:
                a, b = 453.85, 2.1166
            c, d = 24.1670, 2.5334

        elif sc == 'B':
            if total_dist <= 0.2:
                a, b = 90.673, 0.93198
            elif total_dist <= 0.4:
                a, b = 98.483, 0.98332
            else:
                a, b = 109.3, 1.09710
            c, d = 18.333, 1.8096

        elif sc == 'C':
            a, b = 61.141, 0.91465
            c, d = 12.5, 1.0857

        elif sc == 'D':
            if total_dist <= 0.3:
                a, b = 34.459, 0.86974
            elif total_dist <= 1:
                a, b = 32.093, 0.81066
            elif total_dist <= 3:
                a, b = 32.093, 0.64403
            elif total_dist <= 10:
                a, b = 33.504, 0.60486
            elif total_dist <= 30:
                a, b = 36.65, 0.56589
            else:
                a, b = 44.053, 0.51179
            c, d = 8.333, 0.72382

        elif sc == 'E':
            if total_dist <= 0.1:
                a, b = 24.260, 0.83660
            elif total_dist <= 0.3:
                a, b = 23.331, 0.81956
            elif total_dist <= 1:
                a, b = 21.628, 0.75660
            elif total_dist <= 2:
                a, b = 21.628, 0.63077
            elif total_dist <= 4:
                a, b = 22.534, 0.57154
            elif total_dist <= 10:
                a, b = 24.703, 0.50527
            elif total_dist <= 20:
                a, b = 26.970, 0.46713
            elif total_dist <= 40:
                a, b = 35.420, 0.37615
            else:
                a, b = 47.618, 0.29592
            c, d = 6.25, 0.54287

        elif sc == 'F':
            if total_dist <= 0.2:
                a, b = 15.209, 0.81558
            elif total_dist <= 0.7:
                a, b = 14.457, 0.78407
            elif total_dist <= 1:
                a, b = 13.953, 0.68465
            elif total_dist <= 2:
                a, b = 13.953, 0.63227
            elif total_dist <= 3:
                a, b = 14.823, 0.54503
            elif total_dist <= 7:
                a, b = 16.187, 0.46490
            elif total_dist <= 15:
                a, b = 17.836, 0.41507
            elif total_dist <= 30:
                a, b = 22.651, 0.32681
            elif total_dist <= 60:
                a, b = 27.074, 0.27436
            else:
                a, b = 34.219, 0.21716
            c, d = 4.1667, 0.36191

        if total_dist > 0:
            big_theta = 0.017453293 * (c - d * np.log(total_dist))
            sigma_y = 465.11628 * total_dist * np.tan(big_theta)
            sigma_z = min(a * (total_dist) ** b, 5000)

            sigma_y_vals.append(sigma_y)
            sigma_z_vals.append(sigma_z)

    return np.mean(sigma_y_vals), np.mean(sigma_z_vals)


def gpuff(Q: float, stab_class: List[str],
          x_p: float, y_p: float,
          x_r_vec: np.ndarray, y_r_vec: np.ndarray, z_r_vec: np.ndarray,
          total_dist: float,
          H: float, U: float) -> np.ndarray:
    # Conversion factor for methane (kg/m^3 to ppm)
    conversion_factor = 1e6 * 1.524

    # Convert total distance from m to km for stability class calculations
    total_dist_km = total_dist / 1000

    # Get sigma values
    sigma_y, sigma_z = compute_sigma_vals(stab_class, total_dist_km)

    # Calculate concentration using Gaussian puff model
    C = (Q / ((2 * np.pi) ** (3 / 2) * sigma_y ** 2 * sigma_z)) * \
        np.exp(-0.5 * ((x_r_vec - x_p) ** 2 + (y_r_vec - y_p) ** 2) / sigma_y ** 2) * \
        (np.exp(-0.5 * (z_r_vec - H) ** 2 / sigma_z ** 2) + np.exp(-0.5 * (z_r_vec + H) ** 2 / sigma_z ** 2))

    # Convert from kg/m^3 to ppm
    C = C * conversion_factor

    # Replace NaNs with zeros
    C = np.nan_to_num(C, 0)

    return C
