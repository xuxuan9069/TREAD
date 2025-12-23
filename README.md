# TREAD
TREAD: A Transformer-Based Regional Earthquake Early Warning Model with Distance Prediction

## Installation (Anaconda or Docker)
### Step0. Clone the repository
```shell
$ git clone https://github.com/xuxuan/TREAD.git
```

### Step1. Create a new Anaconda virtual environment
```shell
$ conda create --name TREAD python=3.8
$ conda activate TREAD
```

### Step2. Install the dependencies
```shell
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

## Config.json
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
$ python train.py --cofig config.json
```

## Model Evaluate
* You can use ```--test_run``` to test only 10 events.
* You can use ```--val``` to test validation dataset.
```shell
$ python evaluate.py --experiment_path <Checkpoint dir> --head_times 
```

## Model Evaluate Results
```shell
$ python analysis.py --path <Test dataset pkl file path>
```