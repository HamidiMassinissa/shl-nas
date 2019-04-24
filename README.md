# Learning Mobility-Related Human Activities from Sensor-Rich Environments
This repository contains code to reproduce experiments used for our submission to ECML 2019.

## Description
Human activity recognition (HAR) is a key component of context-aware-enabled applications.
This paper explores the suitability of neural architectures for recognizing mobility-related human activities in a fully-featured, sensor and position-rich environment which is provided by the internet of things (IoT).

We define a set of convolutional modes in order to perform sensor and position fusion.
Neural architecture search techniques allow us to tune the learning stages and lead to a robust combination of the hyperparameters.
Besides, exploration of the hyperparameter space allows us to interpret interactions among data sources based on the conjunction of the modality they provide as well as their position.
We report on experiments conducted on the recently released Sussex-Huawei locomotion-transportation (SHL) dataset which provides us with a fully featured environment.

## Getting Started

### Prerequisites
* `numpy`
* `TensorFlow`
* `scikit-optimize`
* `fanova` to install, please follow the steps [here](https://automl.github.io/fanova/install.html)

If you are using `pip` package manager, you can simply install all requirements via the following command(s):

    python -m virtualenv .env -p python3 [optional]
    source .env/bin/activate [optional]
    pip3 install -r requirements.txt

### Installing
#### Get the dataset
1. You can get the preview of the SHL dataset from [here](http://www.shl-dataset.org/download/#shldataset-preview). Make sure to put the downloaded files into `./data/` folder.
2. Run `extract_data.sh` script which will extract the dataset into `./generated/tmp/` folder.

## Running
### Bayesian optimization
In order to run Bayesian optimization, you can issue the following command:

    python3 shl-nas.py --bayesopt
    
Additionally, you can specify a subset of the data generators you want to apply Bayesian optimization on as follows:

    python3 shl-nas.py --bayesopt --position {bag|torso|hand|hips}

### Functional analysis of variance
You can find a complete notebook showing the functional analysis of variance inside `notebooks/` folder.

## Authors (Contact)
* Massinissa Hamidi (hamidi@lipn.univ-paris13.fr)
* Aomar Osmani
