#!/bin/bash

##############################################################################
# RUN EXPERIMENTS ON ALL DATASETS USING THE VGG 16 MODEL & NO DYNAMIC KERNEL #
##############################################################################
# This will run the experiments with the following default arguments:
#   - Batch size:      64
#   - Epochs:         200
#   - Learning rate:    0.1
#   - Kernel size:      3

# For each dataset...
for dataset in cifar10 cifar100 mnist svhn
do
    # run an experiment with no dynamic kernel.
    python main.py \
        --model vgg \
        --dataset $dataset
done