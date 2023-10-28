#!/bin/bash

##########################################################
# RUN EXPERIMENTS ON ALL DATASETS USING THE VGG 16 MODEL #
##########################################################
# This will run the experiments with the following default arguments:
#   - Batch size:      64
#   - Epochs:         200
#   - Learning rate:    0.1
#   - Kernel size:      3
#   - Location:         0
#   - Scale:            1

# For each dataset...
for dataset in cifar10 cifar100 imagenet mnist
do
    # distribution...
    for distribution in cauchy gaussian gumbel laplace poisson
    do
        # and kernel type, ...
        for kernel_type in $(seq 1 14);
        do
            # Clear terminal screen
            clear

            # run an experiment with the normal CNN model.
            python main.py \
                --model resnet \
                --distribution $distribution \
                --kernel_type $kernel_type \
                --dataset $dataset

            git add ./experiments/results.csv
            git add ./logs/*
            git add ./output/*
            git commit -m "$(date +'%F %T'): VGG 16 | $distribution | $kernel_type | $dataset"
            git push origin main
        done
    done
done