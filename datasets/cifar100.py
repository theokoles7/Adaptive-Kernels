"""Cifar 100 dataset and utilities."""

import torch.utils.data as data, typing
from torchvision    import datasets, transforms

from utils          import ARGS, LOGGER

class Cifar100():
    """This dataset is just like the CIFAR-10 
    (https://www.cs.toronto.edu/~kriz/cifar.html), except it has 100 
    classes containing 600 images each. There are 500 training images 
    and 100 testing images per class. The 100 classes in the CIFAR-100 
    are grouped into 20 superclasses. Each image comes with a "fine" 
    label (the class to which it belongs) and a "coarse" label (the 
    superclass to which it belongs). 
    """

    # Initialize logger
    _logger = LOGGER.getChild('cifar100-dataset')

    def __init__(self, path: str = ARGS.dataset_path, batch_size: int = ARGS.batch_size):
        """Initialize Cifar100 dataset loaders.

        Args:
            path (str, optional): Path at which dataset is located/can be downloaded. Defaults to ARGS.dataset_path.
            batch_size (int, optional): Dataset batch size. Defaults to ARGS.batch_size.
        """
        # Create transform
        self._logger.info("Initializing transform")
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))
        ])

        # Verify train data
        self._logger.info("Verifying/downloading train data")
        train_data = datasets.CIFAR100(
            root =          path,
            download =      True,
            train =         True,
            transform =     transform
        )

        # Verify test data
        self._logger.info("Verifying/downloading test data")
        test_data = datasets.CIFAR100(
            root =          path,
            download =      True,
            train =         False,
            transform =     transform
        )

        # Create training loader
        self.logger.info("Creating train data loader.")
        self.train_loader = data.DataLoader(
            train_data,
            batch_size =    batch_size,
            pin_memory =    True,
            num_workers =   4,
            shuffle =       True,
            drop_last =     True
        )

        # Create testing loader
        self.logger.info("Creating test data loader.")
        self.test_loader = data.DataLoader(
            test_data,
            batch_size =    batch_size,
            pin_memory =    True,
            num_workers =   4,
            shuffle =       True,
            drop_last =     False
        )

        # Define parameters
        self.num_classes =  100
        self.channels_in =    3
        self.dim =           32

        self._logger.debug(f"DATASET: {self} | CLASSES: {self.num_classes} | CHANNELS: {self.channels_in} | DIM: {self.dim}")
        self._logger.debug(f"TRAIN LOADER:\n{vars(self.train_loader)}")
        self._logger.debug(f"TEST LOADER:\n{vars(self.test_loader)}")

    def get_loaders(self) -> typing.Tuple[data.DataLoader, data.DataLoader]:
        """Fetch data loaders.

        Returns:
            typing.Tuple[data.DataLoader, data.DataLoader]: Cifar100 train & test loaders
        """
        return self.train_loader, self.test_loader
    
    def __str__(self) -> str:
        """Provide string format of Cifar100 dataset object.

        Returns:
            str: String format of Cifar100 dataset
        """
        return f"Cifar100 dataset ({self.num_classes} classes)"