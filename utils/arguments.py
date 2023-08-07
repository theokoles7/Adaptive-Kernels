"""Command line arguments utilities."""

import argparse

class Arguments():
    """Command line argument parser."""

    def __init__(self):
        """Initalize arguments self.parser and define arguments."""
        # Initialize self.parser
        self.parser = argparse.ArgumentParser(
            prog='hi-lo',
            description='Fork of KernelSet, intended for studying efficacy of models on high or low resolution images'
        )

        # BEGIN ARGUMENTS =============================================================================

        # Universal Arguments -----------------------
        self.parser.add_argument(
            '--verbose',
            action='store_true',
            help='log all steps'
        )

        self.parser.add_argument(
            '--debug',
            action='store_true',
            help='log variables'
        )

        # Dataset Arguments -------------------------
        dataset = self.parser.add_argument_group('Dataset')

        dataset.add_argument(
            '--dataset',
            type=str,
            choices=['caltech', 'cifar10', 'cifar100', 'imagenet', 'mnist', 'svhn'],
            default='cifar10',
            nargs=1,
            help='choice of dataset'
        )

        dataset.add_argument(
            '--dataset_path',
            type=str,
            default='DATA',
            help='path to dataset pool'
        )

        dataset.add_argument(
            '--batch_size', '-bs',
            type=int,
            default=64,
            help='specify batch size (defaults to 64)'
        )

        # Model Arguments ---------------------------
        model = self.parser.add_argument_group('Model')

        model.add_argument(
            '--model', '-M',
            type=str,
            choices=['normal', 'resnet', 'vgg'],
            default='resnet',
            nargs=1,
            help='choice of CNN model'
        )

        model.add_argument(
            '--epochs', '-e',
            type=int,
            default=200,
            help='specify number of epochs (defaults to 200)'
        )

        model.add_argument(
            '--epoch_limit',
            type=int,
            default=200,
            help='epoch limit'
        )

        model.add_argument(
            '--learning_rate', '-lr',
            type=int,
            default=1e-1,
            help='specify learning rate (defaults to 0.1)'
        )
        
        model.add_argument(
            '--kernel_size', 
            type=int,
            default=3,
            help='kernel size (square)'
        )

        model.add_argument(
            '--save_params',
            action='store_true',
            help='save model parameters'
        )

        # Distribution Arguments --------------------
        distro = self.parser.add_argument_group("Distribution")

        distro.add_argument(
            '--distribution', '-D',
            type=str,
            choices=['cauchy', 'gaussian', 'gumbel', 'laplace', 'poisson'],
            default='gaussian',
            help='choice of probability distribution'
        )

        # Univariate
        distro.add_argument(
            '--rate',
            type=float,
            default=0,
            help='distribution rate parameter for a univariate distribution (Poisson)'
        )

        # Bivariate
        distro.add_argument(
            '--location',
            type=float,
            default=0,
            help='distribution location parameter for a bivariate distribution (Gaussian, Cauchy, Gumbel, Laplace)'
        )

        distro.add_argument(
            '--scale',
            type=self._check_scale,
            default=1,
            help='distribution scale parameter for a bivariate distribution (Gaussian, Cauchy, Gumbel, Laplace) (greater than zero)'
        )

        distro.add_argument(
            '--kernel_type',
            type=int,
            default=13,
            help='kernel configuration type'
        )

        # Logger Arguments --------------------------
        logger = self.parser.add_argument_group('Logger')

        logger.add_argument(
            '--logger_path',
            type=str,
            default='LOGS',
            help='specify log output path'
        )

        # Output Arguments --------------------------
        output = self.parser.add_argument_group('Output')

        output.add_argument(
            '--output_path',
            type=str,
            default='OUTPUT',
            help='specify output data path'
        )

        # END ARGUMENTS ===============================================================================

    def get_args(self) -> argparse.Namespace:
        """Parse command line arguments and provide values.

        Returns:
            argparse.Namespace: Argparse NameSpace of arguments values
        """
        # Return argument values
        return self.parser.parse_args()

    def _check_scale(self, scale: str) -> float:
        """Verify that provided scale argument is within range [0, inifinity)

        Args:
            scale (str): Provided scale

        Returns:
            float: Float value of scale

        Raises:
            argparse.ArgumentTypeError: If scale is less than or equal to zero
        """
        if float(scale) <= 0: raise argparse.ArgumentTypeError('Scale must be greater than zero')
        return float(scale)
    
    def __str__(self) -> str:
        """Provide string format of Arguments object.

        Returns:
            str: String format
        """
        arg_string = "ARGUMENTS"

        for arg in vars(self.get_args()):
            arg_string += f"\n\t{arg:<20}{(vars(self.get_args())[arg]):>20}"

        return arg_string