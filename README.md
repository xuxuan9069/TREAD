# TREAD
TREAD: A Transformer-Based Regional Earthquake Early Warning Model with Distance Prediction

## Installation (Anaconda or Docker)
### Step0. Clone the repository
```shell
$ git clone https://github.com/xuxuan9069/TREAD.git
```

### Step1. Create a new Anaconda virtual environment
```shell
$ conda create --name TREAD python=3.8
$ conda activate TREAD
```

### Step2. Install the dependencies
```shell
$ cd TREAD
$ pip install -r requirements.txt
```

### (Optional) Create environment through Docker
#### Build the docker image
```shell
$ docker build -t TREAD:v1 .
```
#### Run the docker container
```shell
$ docker run --rm -v $(pwd):/TREAD TREAD:v1
```

## Dataset
The original dataset is stored as a large multi-event HDF5 file. 
Due to file size constraints, this repository provides a single-event HDF5 file as a structural example only.
The example file(20201210131958659999.hdf5) preserves exactly the same internal hierarchy as the original dataset.
### HDF5 Data Structure
```text
/
├── metadata
│   ├── channels
│   ├── sampling_rate
│   ├── time_before
│   ├── time_after
│   ├── pga_thresholds
│   └── event_metadata
│       ├── source_latitude_deg
│       ├── source_longitude_deg
│       ├── source_depth_km
│       ├── source_magnitude
│       ├── source_origin_time
│       └── data_file
│
└── data
    ├── <event_id_1>
    │   ├── waveforms
    │   ├── p_picks
    │   ├── pga
    │   ├── pga_times
    │   ├── pgv
    │   ├── pgv_times
    │   ├── coords
    │   ├── stations
    │   └── trace_filename
    │
    ├── <event_id_2>
    │   └── ...
    │
    └── <event_id_N>
        └── ...
```
* event_id is the event start time.

## Setting
* The settings are in config.json.
* model_params:
  - **distance prediction module**: ```distance_type``` (v1(SC)/v2(BI))
* training_params:
  - **GPU ID Setting**: ```device```
  - **Dataset**: ```train_data_path```, ```val_data_path```, ```test_data_path```
  - **Checkpoint dir**: ```weight_path```
  - **Distance/PGA loss ratio**: ```loss_ratio```

## Model Training
* You can use ```--test_run``` to test only 10 events.
```shell
$ python train.py --config config.json
```

## Model Evaluate
* You can use ```--test_run``` to test only 10 events.
* Evaluate validation and test dataset.
```shell
$ python evaluate.py --experiment_path TREAD_SC_0.0025 --head_times --val
```
```shell
$ python evaluate.py --experiment_path TREAD_SC_0.0025 --head_times 
```

## Model Alert Performance Results
```shell
$ python analysis.py --path TREAD_SC_0.0025/evaluation/test/head_times_predictions.pkl
```