#!/bin/bash

##############################################################
# RUN JOBS ON ALL MODELS AND KERNELS USING THE MNIST DATASET #
##############################################################
# This will run the experiments with the following default arguments:
#   - Batch size:       64
#   - Epochs:          200
#   - Learning Rate:     0.1
#   - Kernel Size:       3

# For each model...
for model in normal resnet vgg
do
    # kernel...
    for distribution in cauchy gaussian gumbel laplace
    do
        # and kernel type...
        for kernel_type in $(seq 1 14);
        do
            # Clear the terminal
            clear

            # Run an experiment with the MNIST dataset
            python main.py run-job $model \
                mnist \
                --kernel_type $kernel_type \
                $distribution

            # Push logs and output to repository
            git add ./logs/*
            git add ./output/*
            git commit -m "$(date +'%F %T'): $model | mnist | $distribution (Type $kernel_type)"
            git push origin main

        done
    done
done