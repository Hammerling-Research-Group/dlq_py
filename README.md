# dlq_py
*Production code for the Python implementation of the DLQ algorithm*

[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)

This repo contains the production code for the Python implementation of the DLQ algorithm. The original version is in R and can be found [here](https://github.com/Hammerling-Research-Group/dlq). As with the R version, the algorithm proceeds in two main steps in an effort to estimate methane emission start and end time (detection), source location (localization), and emission rate (quantification) using concentration observations from a network of point-in-space continuous monitoring systems. 

For more on this work, see Daniels et al. (2024), https://doi.org/10.1525/elementa.2023.00110

## Notes on the Python Implementation

Here's the current structure (*subject to change as this project is in active development*):

```bash
dlq_py/
├── code/
│   ├── step1_simulate.py
│   ├── step2_dlq.py
│   ├── helper_distance_conversions.py
│   ├── helper_gpuff_function.py
│   ├── Input_data.zip
│   ├── Input_data_step2.zip
├── tests/
│   ├── test_step1_simulate.py
│   ├── test_step2_dlq.py
├── LICENSE
├── README.md
└── env.yml
```

A few key notes are worth highlighting for the Python implementation: 

  - Before using the scripts, users need to have a properly configured environment. This can be loaded, to ensure all needed dependencies are available, by running `conda env create -f env.yml`.
  - Execute step 1 by running `python step1_simulate.py`
  - Step 2 processes raw sensor readings by removing background noise, detecting spikes in methane concentrations (events), aligning the detected events with simulation data, and performing a quantitative analysis of these events, including emission source localization and rate estimation
  - For step 2, be sure to edit the configuration file to adjust parameters such as file paths, thresholds for event detection, and simulation data location, e.g.,

```python
config = {
    'gap_time': 5,
    'length_threshold': 2,
    'simulation_data_path': "/path/to/simulation_output_new.pkl",
    'output_file_path': "/path/to/output/",
}
```

  - Execute step 2 by running `python step2_dlq.py`

## Installation

Though the current code is still largely in "research code" form, users are still encouraged to engage with it. 

To do so, the simplest approach is to clone the repo and work from the `dlq_py` directory. 

1. Set your desired directory from which to work. E.g., for your Desktop:

```bash
$ cd Desktop
```

2. Clone and store `dlq_py` at the desired location:

```bash
$ git clone https://github.com/Hammerling-Research-Group/dlq_py.git
```

3. Move into the cloned `dlq_py` directory:

```bash
$ cd dlq_py
```

## Usage

  1. Place the input CSV files in the same directory where the script is executed or update the file paths in the script accordingly
  2. Run the script for step 1 (`step1_simulate.py`)
  3. Store the output
  4. Run the script for step 2 (`step2_dlq.py`)
  5. Store the output

## Example

### Step 1

Example Input:
  - **Sensor Data CSV**: Timestamps and methane concentrations from different sensor locations.
  - **Source Locations CSV**: Latitude and longitude coordinates of emission sources.
  - **Receptor Locations CSV**: Latitude and longitude of receptor locations where methane concentrations will be computed.
  
Example Output:
  - **aligned_data.csv**: Cleaned and time-aligned data from sensors and wind sources.
  - **simulation_output.pkl**: Pickle file with serialized simulation results.
  - **final_concentrations.csv**: CSV with predicted methane concentrations at receptor locations.

### Step 2

  1.	Initial Data Loading: Load raw sensor data and simulation outputs.
  2.	Background Removal: Clean up the data by removing long-term background methane levels.
  3.	Event Detection: Automatically detect and classify events based on concentration spikes.
  4.	Event-Simulation Alignment: Compare detected events with simulation data to ensure consistency and calculate performance metrics.
  5.	Source Localization and Emission Rate Estimation: Estimate the location of methane sources and quantifies emission rates for each detected event.

