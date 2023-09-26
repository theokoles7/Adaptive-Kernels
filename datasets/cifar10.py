"""Cifar 10 utilities."""

import typing

from termcolor import colored
import torch.utils.data as data
from torchvision import transforms, datasets

from utils import ARGS, LOGGER

class Cifar10():
    """The CIFAR-10 dataset (https://www.cs.toronto.edu/~kriz/cifar.html) consists 
    of 60000 32x32 colour images in 10 classes, with 6000 images 
    per class. There are 50000 training images and 10000 test images.
    """
    # Initialize logger
    logger = LOGGER.getChild('cifar10-dataset')

    def __init__(self, path: str, batch_size: int):
        """Initialize Cifar10 dataset object.

        Args:
            path (str): Dataset path
            batch_size (int): Dataset batch size
        """
        # Create transform for loaders
        self.logger.info("Initializing transform.")
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))
        ])

        # Verify train data
        self.logger.info("Verifying/downloading train data.")
        train_data = datasets.CIFAR10(
            root =          path,
            download =      True,
            train =         True,
            transform =     transform
        )

        # Verify test data
        self.logger.info("Verifying/downloading test data.")
        test_data = datasets.CIFAR10(
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
        self.num_classes =  10
        self.channels_in =   3
        self.dim =          32

        if ARGS.debug:
            self.logger.debug(f"DATASET: {self}")
            self.logger.debug(f"TRAIN LOADER: \n{vars(self.train_loader)}")
            self.logger.debug(f"TEST LOADER: \n{vars(self.test_loader)}")
            self.logger.debug(f"CLASSES: {self.num_classes}, CHANNELS: {self.channels_in}, DIM: {self.dim}")

    def get_loaders(self) -> typing.Tuple[data.DataLoader, data.DataLoader]:
        """Fetch data loaders.

        Returns:
            typing.Tuple[data.DataLoader, data.DataLoader]: Cifar10 train loader, test loader
        """
        return self.train_loader, self.test_loader
    
    def __str__(self) -> str:
        """Provide string format of class.

        Returns:
            str: String format of Cifar10 dataset
        """
        return f"Cifar10 dataset ({self.num_classes} classes)"
    
if __name__ == '__main__':
    """Test dataset."""

    # Initialize dataset
    dataset = Cifar10('./data', 64)

    # Print dataset __str__
    print(f"\n{colored('DATASET', 'magenta')}: \n{dataset}")

    # Print train loader
    print(f"\n{colored('TRAIN LOADER', 'magenta')}: \n{vars(dataset.train_loader)}")

    # Print test loader
    print(f"\n{colored('TEST LOADER', 'magenta')}: \n{vars(dataset.test_loader)}")