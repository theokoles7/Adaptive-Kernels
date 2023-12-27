"""MNIST dataset and utilities."""

import torch.utils.data as data, typing
from torchvision    import datasets, transforms

from utils          import ARGS, LOGGER

class MNIST():
    """The MNIST (http://yann.lecun.com/exdb/mnist/) database of 
    handwritten digits, available from this page, has a training set of 
    60,000 examples, and a test set of 10,000 examples. It is a 
    subset of a larger set available from NIST. The digits have been 
    size-normalized and centered in a fixed-size image.
    """

    # Initialize logger
    _logger = LOGGER.getChild('mnist-dataset')

    def __init__(self, path: str = ARGS.dataset_path, batch_size: int = ARGS.batch_size):
        """Initialize MNIST dataset loaders.

        Args:
            path (str, optional): Path at which dataset is located/can be downloaded. Defaults to ARGS.dataset_path.
            batch_size (int, optional): Dataset batch size. Defaults to ARGS.batch_size.
        """
        # Create transform for loaders
        self._logger.info("Initializing transform.")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Verify train data
        self._logger.info("Verifying/downloading train data.")
        train_data = datasets.MNIST(
            root =          path,
            download =      True,
            train =         True,
            transform =     transform
        )

        # Verify test data
        self._logger.info("Verifying/downloading test data.")
        test_data = datasets.MNIST(
            root =          path,
            download =      True,
            train =         False,
            transform =     transform
        )

        # Create training loader
        self._logger.info("Creating train data loader.")
        self.train_loader = data.DataLoader(
            train_data,
            batch_size =    batch_size,
            pin_memory =    True,
            num_workers =   4,
            shuffle =       True,
            drop_last =     True
        )

        # Create testing loader
        self._logger.info("Creating test data loader.")
        self.test_loader = data.DataLoader(
            test_data,
            batch_size =    batch_size,
            pin_memory =    True,
            num_workers =   4,
            shuffle =       True,
            drop_last =     False
        )

        # Define parameters
        self.num_classes =  10
        self.channels_in =   1
        self.dim =          16

        self._logger.debug(f"DATASET: {self} | CLASSES: {self.num_classes} | CHANNELS: {self.channels_in} | DIM: {self.dim}")
        self._logger.debug(f"TRAIN LOADER:\n{vars(self.train_loader)}")
        self._logger.debug(f"TEST LOADER:\n{vars(self.test_loader)}")

    def get_loaders(self) -> typing.Tuple[data.DataLoader, data.DataLoader]:
        """Fetch data loaders.

        Returns:
            typing.Tuple[data.DataLoader, data.DataLoader]: MNIST train & test loaders
        """
        return self.train_loader, self.test_loader
    
    def __str__(self) -> str:
        """Provide string format of MNIST dataset object.

        Returns:
            str: String format of MNIST dataset
        """
        return f"MNIST dataset ({self.num_classes} classes)"