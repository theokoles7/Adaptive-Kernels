Adaptive Kernel Selection for Convolutional Neural Network
#Paper Under Review ICLR 2023

The official PyTorch implementation of Adaptive Kernel Selection for Convolutional Neural Networke (ICLR 2023 #Paper Under Review#).

Required Packages:

python 3.6.3

numpy 1.19.2

pandas 1.1.5

pytorch 1.3.1

scipy 1.5.2

seaborn 0.11.1

termcolor 1.1.0

torchvision

scikit-learn

USAGE: Simply use the code by running:

python3 main.py --dataset <DATASET> --alg <MODEL> --data <PATH_TO_DATA> --kernel_type <Kernel_Type>

For example, to train a ResNet on CIFAR10 basen on Kernel Type 1 and the data is saved in ./data/, we can run:

python3 main.py --dataset cifar10 --alg res --data ./data/ --kernel_type 1
