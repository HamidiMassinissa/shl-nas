# Domain Models-Based Data Sources Integration for HAR

## Description
<img align="right" width="500" src="https://user-images.githubusercontent.com/49691301/64601419-2e440480-d3bd-11e9-90cb-3e0df329fdf1.png">
The ever-increasing quantities of data generated by internet of things applications bring diverse and rich perspectives about monitored phenomena.
This is the case, for example, of applications monitoring continuously human activities from wearable sensor-deployments for diverse purposes like eHealth.
However, this poses critical issues as of how to manage and process all these sources of information efficiently.
Additional knowledge about, *e.g.* the structure of the sensors deployments, the dynamics of human activities, physical models of the body movements, *etc.*, in short, domain models, have the potential to bring efficiency and robustness to the process of data sources integration.
In this contribution, we propose to leverage domain models in order to integrate information sources efficiently.
Precisely, (1) a model of the influence of data sources w.r.t. particular human activities, (2) a model of the interactions between data sources, and (3) a model of the transitions between human activities are composed in order to sample, from the data sources, information to fuse.
The proposed approach is able to perform intelligent data sampling from information sources by taking into consideration the infrastructure of the sensor-rich environments and their evolution.
Experimental evaluations show that the proposed approach ensures (1) an efficient trade-off between the quantities of sampled data and recognition performances by outperforming the baseline setting which exploits all available data and (2) improves robustness towards activities transitions.

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
1. You can get the preview of the SHL dataset (`tar.gz` files) from [here](http://www.shl-dataset.org/download/#shldataset-preview). Make sure to put the downloaded files into `./data/` folder.
2. Run `extract_data.sh` script which will extract the dataset into `./generated/tmp/` folder.

## Running
### Bayesian optimization
In order to run Bayesian optimization, you can issue the following command:

    python3 shl-nas.py --run bayesopt
    
Additionally, you can specify a subset of the data generators you want to apply Bayesian optimization on as follows:

    python3 shl-nas.py --run bayesopt --position {bag|torso|hand|hips}

### Functional analysis of variance
You can find a complete notebook showing the functional analysis of variance inside `notebooks/` folder.

### Training a single model
Similarly, you can train a single model by issuing the following command:

    python3 shl-nas.py --run trainSingleModel [--position {bag|torso|hand|hips}]

## Authors (Contact)
* Massinissa Hamidi (hamidi@lipn.univ-paris13.fr)
* Aomar Osmani
