#!/bin/bash

##########################################################################
# RUN EXPERIMENTS ON ALL DATASETS USING THE VGG 16 MODEL & CAUCHY KERNEL #
##########################################################################
# This will run the experiments with the following default arguments:
#   - Batch size:      64
#   - Epochs:         200
#   - Learning rate:    0.1
#   - Kernel size:      3
#   - Location:         0
#   - Scale:            1

# For each dataset...
for dataset in cifar10 cifar100 mnist svhn
do
    # and kernel type, ...
    for kernel_type in $(seq 1 14);
    do
        # run an experiment with the Cauchy kernel.
        python main.py \
            --model vgg \
            --distribution cauchy \
            --kernel_type $kernel_type \
            --dataset $dataset
    done
done