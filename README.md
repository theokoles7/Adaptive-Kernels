# Efficacy of Probability Distributions Against High/Low Resolution Images
Work based on *Adaptive Kernel Selection* (Dr. Sahan Ahmad)

## Prerequisites:
| Package       | Version   |
|---------------|-----------|
| python        | >= 3.6.3  |
| numpy         | >= 1.19.2 |
| pandas        | >= 1.1.5  |
| pytorch       | >= 1.3.1  |
| scipy         | >= 1.5.2  |
| seaborn       | >= 0.11.1 |
| termcolor     | >= 1.1.0  |
| torchvision   | N/A       |
| scikit-learn  | N/A       |
| tqdm          | N/A       |

To install the packages required for this application, run the following command(s) from the project's root directory: `conda env create -f hilo_env.yaml && conda activate hilo`

## USAGE: 
Running the application is as simple as:

`python main.py --dataset <DATASET> --alg <MODEL> --data <PATH_TO_DATA> --kernel_type <Kernel_Type>`

For example, to train a ResNet on CIFAR10 basen on Kernel Type 1 and the data is saved in ./data/, we can run:

`python main.py --dataset cifar10 --alg res --data ./data/ --kernel_type 1`

# Background & Related Work

## Distributions Studied

### Gaussian (Normal) Distribution

### Poisson Distribution

### LaPlace Distribution