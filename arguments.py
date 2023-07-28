"""Parse and handle command line arguments."""

import argparse, torch

def get_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Arguments Namespace
    """
    # Initialize parser
    parser = argparse.ArgumentParser(
        prog = 'hilo',
        description = 'Fork of KernelSet, intended for studying efficacy of models on high or low resolution images'
    )

    # ARGUMENTS -----------------------------------------------------------------------------------
    parser.add_argument(
        '--data', 
        type = str, 
        default = '../data',
        help = 'data source/location of datasets'
    )

    parser.add_argument(
        '--dataset',
        type = str,
        choices = [
            'mnist', 'cifar10', 'cifar100', 
            'imagenet', 'svhn', 'caltech'
        ],
        default = 'cifar10',
        help = 'target dataset'
    )

    parser.add_argument(
        '--model', 
        type = str, 
        choices = ['normal', 'vgg', 'res'],
        default = 'res',
        help = 'choice of model'
    )

    parser.add_argument(
        '--log_name', 
        type = str, 
        default = 'cbs',
        help = 'name of logger'
    )

    parser.add_argument(
        '--log_path', 
        type = str, 
        default = 'logs',
        help = 'path at which log file will be created'
    )

    parser.add_argument(
        '--no-cuda',
        action = 'store_true',
        help = 'opt out of running on GPU'
        )

    parser.add_argument(
        '--batch_size', 
        type = int, 
        default = 64,
        help = 'specify batch size'
    )

    parser.add_argument(
        '--num_epochs', 
        type = int, 
        default = 200,
        help = 'specify number of epochs'
    )

    parser.add_argument(
        '--lr', 
        type = float, 
        default = 1e-1,
        help = 'specify learning rate'
    )

    parser.add_argument(
        '--ssl', 
        action = 'store_true',
        help = 'SSL *****'
    )

    parser.add_argument(
        '--percentage', 
        type = int, 
        default = 10,
        help = "percentage *****"
    )

    parser.add_argument(
        '--save_model', 
        action='store_true',
        help = 'save model parameters'
    )


    # CBS ARGS
    parser.add_argument(
        '--std', 
        type = float,
        default = 1,
        help = 'standard deviation'
    )

    parser.add_argument(
        '--std_factor',
        type = float,
        default = 0.9,
    )

    parser.add_argument(
        '--epoch', 
        type = int,
        default = 5, 
    )
    
    parser.add_argument(
        '--kernel_size', 
        type=int,
        default=3,
        help = 'kernel size (square)'
    )

    # DADL ARGS
    parser.add_argument(
        '--epoch_limit',
        type=int,
        default = 200
    )

    parser.add_argument(
        '--precision_point', 
        type = int, 
        default = 2
    )

    # 2Kernel Set ARGS
    parser.add_argument(
        '--kernel_type', 
        type = int,
        default = 13,
        help = 'kernel configuration type'
    )
    # ---------------------------------------------------------------------------------------------

    args = parser.parse_args()

    args.cuda = (not args.no_cuda and torch.cuda.is_available())

    return args

