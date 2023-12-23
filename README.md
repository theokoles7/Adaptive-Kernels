# Efficacy of Probability Distribution Against High/Low Resolution Images
Work based on *Adaptive Kernel Selection* (Dr. Sahan Ahmad)

## Table of Contents
* [Prerequisites](#prerequisites)
* [Operating Application](#operating-application)
* [Modules/Components](#modulescomponents)

## Prerequisites

### Python
If you do not already have Python 3.10 or higher installed, do so [here](https://www.python.org/downloads/).

### Anaconda
If you do not already have Anaconda installed, you can find out how to do so [here](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html).

### Create Conda Environment
Within the `conf/` directory, there is already an environment defined with all requirements. To create/activate it, simply run:
```
conda env create -f conf/hilo_env.yaml  # Creation
conda activate hilo                     # Activation
```

## Operating Application

### Commands

#### Initialize Results
When jobs/experiments are run, the results will be recorded automatically. However, in order for this to work, a CSV file containing combinations of models, datasets, distributions, and kernel types must be initialized *prior to running the job/experiment*, using the following command:
```
python main.py init-results
```

#### Run Job
Run any individual job with the following command:
```
python main.py run-job {MODEL} {DATASET} [--kernel_type {KERNEL_TYPE}] [OPTIONS] [DISTRIBUTION]
```
* Model choices: `normal`, `resnet`, or `vgg`
* Dataset choices: `cifar10`, `cifar100`, `imagenet`, `mnist`
* Kernel distribution choices: `cauchy`, `gaussian`, `gumbel`, `laplace`, `poisson`
    * Jobs can be run with no kernel
    * For each kernel, there are additional options:
        * `--location`: Distribution location parameter (Cauchy, Gaussian, Gumbel, & Laplace)
        * `--scale`: Distribution scale parameter (Cauchy, Gaussian, Gumbel, & Laplace)
        * `--rate`: Poisson rate parameter
* Job options:
    * Model:
        * `--learning_rate`, `-lr`: Optimizer learning rate. Defaults to 0.1.
        * `--save_params`: Save model parameters on job completion.
    * Dataset:
        * `--dataset_path`: Directory at which datasets will be downloaded/pulled. Defaults to `./data/`.
        * `--batch_size`: Dataset batch size. Defaults to 64.
    * Kernel:
        * `--kernel_size`: Kernel size (square). Defaults to 3.
        * `--kernel_type`: Kernel configuration type. Defaults to 13 (Read *Adaptive Kernel Selection* by Dr. Sahan Ahmad to find out why).


#### Run Experiment
Run any experiment set with the following command:
```
python main.py run-experiment [--control] [--by_model {MODEL}] [--by_dataset {DATASET}] [--by_kernel {DISTRIBUTION}]
```
Experiment can be run using the `run-experiment` command, using one of the following options/flags:
* `--control`: Runs control experiments (no argument required, as this is a simple flag)
* `--by_model`: Runs experiments with choice of model (choices are `normal`, `resnet`, or `vgg`)
* `--by_dataset`: Runs experiments with choice of dataset (choices are `cifar10`, `cifar100`, `imagenet`, `mnist`)
* `--by_kernel`: Runs experiments with choice of kernel distribution (choices are `cauchy`, `gaussian`, `gumbel`, `laplace`, `poisson`)

### Universal Options

#### Logging
* `logging_path`: Directory at which log files will be written. Defaults to `./logs/`
* `logging_level`: Minmum logging level (DEBUG < INFO < WARNING < ERROR < CRITICAL). Defaults to INFO.

#### Output
* `output_path`: Directory at which output/results files will be written. Defaults to `./output/`

## Modules/Components

### Probability Distributions

#### Cauchy

#### Gaussian

#### Gumbel

#### Laplace

#### Poisson

### Datasets

#### Cifar-10

#### Cifar-100

#### ImageNet

#### MNIST

### Models

#### Basic CNN

#### ResNet 18

#### VGG 16