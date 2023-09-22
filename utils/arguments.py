"""Command line argument utilities."""

import argparse

class Arguments():
    """Command line argument parser."""

    def __init__(self):
        """Initialize argument parser and define arguments."""
        # Initialize parser
        self._parser = argparse.ArgumentParser(
            prog =          'hi-lo',
            description =   (
                            'For of KernelSet, inteded for studying '
                            'the efficacy of models on high and low '
                            'resolution images, using various '
                            'probability distributions as kernels.'
                            )
        )

        # BEGIN ARGUMENTS =========================================================================

        # Universal -----------------------------

        # Dataset -------------------------------
        dataset = self._parser.add_argument_group('Dataset')

        dataset.add_argument(
            '--dataset',
            type =      str,
            choices =   ['cifar10', 'cifar100', 'mnist', 'svhn'],
            default =   'cifar10',
            help =      'choice of dataset (defaults to cifar10)'
        )

        dataset.add_argument(
            '--dataset_path',
            type =      str,
            default =   'data',
            help =      'path at which dataset will be downloaded/loaded (defaults to ./data/)'
        )

        dataset.add_argument(
            '--batch_size',
            type =      int,
            default =   64,
            help =      'specify batch size (defaults to 64)'
        )

        # Model ---------------------------------
        model = self._parser.add_argument_group('Model')

        model.add_argument(
            '--model', '-M',
            type =      str,
            choices =   ['normal', 'resnet', 'vgg'],
            default =   'normal',
            help =      'choice of CNN model (defaults to Normal CNN)'
        )

        model.add_argument(
            '--epochs', '-e',
            type =      int,
            default =   200,
            help =      'specify number of epochs (defaults to 200)'
        )

        model.add_argument(
            '--epoch_limit',
            type =      int,
            default =   200,
            help =      'epoch limit (defaults to 200)'
        )

        model.add_argument(
            '--learning_rate', '-lr',
            type =      int,
            default =   1e-1,
            help =      'specify learning rate (defaults to 0.1)'
        )

        model.add_argument(
            '--kernel_size',
            type =      int,
            default =   3,
            help =      'kernel size (square) (defaults to 3)'
        )

        model.add_argument(
            '--save_params',
            action =    'store_true',
            help =      'save model parameters'
        )

        # Kernel --------------------------------
        kernel = self._parser.add_argument_group('Kernel')

        kernel.add_argument(
            '--distribution',
            type =      str,
            choices =   ['cauchy', 'gaussian', 'gumbel', 'laplace', 'poisson'],
            default =   None,
            help =      'choice of probability distribution (defaults to None)'
        )

        kernel.add_argument(
            '--rate',
            type =      float,
            default =   0.,
            help =      'distribution rate parameter (Poisson) (defaults to 0)'
        )

        kernel.add_argument(
            '--location',
            type =      float,
            default =   0.,
            help =      'distribution location parameter (Cauchy, Gaussian, Gumbel, Laplace) (defaults to 0)'
        )

        kernel.add_argument(
            '--scale',
            type =      float,
            default =   1.,
            help =      'distribution scale parameter (Cauchy, Gaussian, Gumbel, Laplace) (defaults to 1)'
        )

        kernel.add_argument(
            '--kernel_type',
            type =      int,
            default =   13,
            help =      'kernel configuration type (defaults to 13)'
        )

        # Logging -------------------------------
        logging = self._parser.add_argument_group('Logging')

        logging.add_argument(
            '--logger_path',
            type =      str,
            default =   'logs',
            help =      'specify log output path (defaults to ./logs/)'
        )

        # Output --------------------------------
        output = self._parser.add_argument_group('Output')

        output.add_argument(
            '--output_path',
            type =      str,
            default =   'output',
            help =      'specify output data path (defaults to ./output/)'
        )

        # END ARGUMENTS ===========================================================================

    def get_args(self) -> argparse.Namespace:
        """Parse command line arguments and provide values in a NameSpace.

        Returns:
            argparse.Namespace: Argparse NameSpace of argument values
        """
        return self._parser.parse_args()