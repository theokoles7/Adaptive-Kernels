#!/bin/bash

##################################################################
# RUN NORMALL CNN ON EACH DATASET, DISTRIBUTION, AND KERNEL TYPE #
##################################################################

# For each dataset, ...
for dataset in cifar10 cifar100 mnist svhn
do
    # distribution, ...
    for distribution in cauchy gaussian gumbel laplace poisson
    do
        # and kernel type, ...
        for kernel_type in $(seq 1 14);
        do
            # Run an experiment.
            python main.py \
                --model vgg \
                --dataset $dataset \
                --distribution $distribution \
                --kernel_type $kernel_type
        done
    done
done
