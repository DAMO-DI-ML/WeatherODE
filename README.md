# WeatherODE

Peiyuan Liu, Tian Zhou, Liang Sun, Rong Jin, "Mitigating Time Discretization Challenges with WeatherODE: A Sandwich Physics-Driven Neural ODE for Weather Forecasting". [[paper](https://arxiv.org/abs/2410.06560)]

## Overview

WeatherODE is a comprehensive framework designed for global and regional weather forecasting based on the ERA5 dataset. The package includes preprocessing scripts, model training pipelines, and evaluation tools tailored for different forecasting horizons (6h, 12h, 18h, and 24h). It supports global and regional forecasting for various areas including Australia, North America, and South America.

## Installation

### Setting Up the Environment

To get started, create and activate a conda environment using the provided configuration:

```bash
conda env create -f environment.yml
conda activate weatherode
```

### Installing WeatherODE

Install the WeatherODE package in editable mode:

```bash
pip install -e .
```

## Data Preparation

### Download ERA5 Data

Download the ERA5 reanalysis dataset from the [WeatherBench](https://dataserv.ub.tum.de/index.php/s/m1524895). Organize the data directory as follows:

```
5.625deg
   ├── 10m_u_component_of_wind
   ├── 10m_v_component_of_wind
   ├── 2m_temperature
   ├── constants.nc
   ├── geopotential
   ├── relative_humidity
   ├── specific_humidity
   ├── temperature
   ├── toa_incident_solar_radiation
   ├── total_precipitation
   ├── u_component_of_wind
   └── v_component_of_wind
```

### Preprocessing

Convert the raw NetCDF files into smaller, more manageable NumPy files and compute essential statistical measures. Execute the following script:

```bash
python src/data_preprocessing/nc2np_equally_era5.py \
    --root_dir /mnt/data/5.625deg \
    --save_dir /mnt/data/5.625deg_npz \
    --start_train_year 1979 --start_val_year 2016 \
    --start_test_year 2017 --end_year 2019 --num_shards 8
```

The preprocessed data directory will have the following structure:

```
5.625deg_npz
   ├── train
   ├── val
   ├── test
   ├── normalize_mean.npz
   ├── normalize_std.npz
   ├── lat.npy
   └── lon.npy
```

## Training

### Global Forecasting

To train a global forecasting model with a 6-hour prediction horizon, use the following command:

```bash
bash ./scripts/global/train_6h.sh
```

Scripts for 12-hour, 18-hour, and 24-hour forecast models are also available.

### Regional Forecasting

For regional forecasting in Australia with a 6-hour prediction horizon, use the following command:

```bash
bash ./scripts/regional/Australia/train_6h.sh
```

Scripts are also provided for 12-hour, 18-hour, and 24-hour forecasts. Additional regions include North America and South America.


## Acknowledgements

We acknowledge the use of the ERA5 reanalysis data provided by the European Centre for Medium-Range Weather Forecasts (ECMWF) and the WeatherBench dataset for benchmarking.

## Citation
If you find this repo useful, please cite our paper.
```
@article{liu2024mitigating,
  title={Mitigating Time Discretization Challenges with WeatherODE: A Sandwich Physics-Driven Neural ODE for Weather Forecasting},
  author={Liu, Peiyuan and Zhou, Tian and Sun, Liang and Jin, Rong},
  journal={arXiv preprint arXiv:2410.06560},
  year={2024}
}
```