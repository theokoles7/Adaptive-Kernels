#!/bin/bash

#######################################################
# RUN JOBS ON ALL MODELS AND DATASETS USING NO KERNEL #
#######################################################
# This will run the experiments with the following default arguments:
#   - Batch size:       64
#   - Epochs:          200
#   - Learning Rate:     0.1
#   - Kernel Size:       3

# For each model...
for model in normal resnet vgg
do
    # and dataset...
    for dataset in cifar10 cifar100 imagenet mnist
    do 
        # Clear the terminal
        clear

        # Run an experiment with no kernel
        python main.py $model $dataset

        # Push logs and output to repository
        git add ./logs/*
        git add ./output/*
        git commit -m "$(date +'%F %T'): $model | $dataset | No distribution kernel"
        git push origin main

    done
done