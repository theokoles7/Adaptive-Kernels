#!/bin/bash

#################################################################
# RUN JOBS ON ALL MODELS AND DATASETS USING THE GAUSSIAN KERNEL #
#################################################################
# This will run the experiments with the following default arguments:
#   - Batch size:       64
#   - Epochs:          200
#   - Learning Rate:     0.1
#   - Kernel Size:       3

# For each model...
for model in normal resnet vgg
do
    # dataset...
    for dataset in cifar10 cifar100 imagenet mnist
    do 
        # and kernel type...
        for kernel_type in $(seq 1 14);
        do
            # Clear the terminal
            clear

            # Run an experiment with the Gaussian kernel
            python main.py run-job $model \
                $dataset \
                --kernel_type $kernel_type
                gaussian

            # Push logs and output to repository
            git add ./logs/*
            git add ./output/*
            git commit -m "$(date +'%F %T'): $model | $dataset | gaussian (Type $kernel_type)"
            git push origin main

        done
    done
done