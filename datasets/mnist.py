"""MNIST utilities."""

import typing

from termcolor import colored
import torch.utils.data as data
from torchvision import transforms, datasets

from utils.logger import LOGGER

class MNIST():
    """The MNIST (http://yann.lecun.com/exdb/mnist/) database of 
    handwritten digits, available from this page, has a training set of 
    60,000 examples, and a test set of 10,000 examples. It is a 
    subset of a larger set available from NIST. The digits have been 
    size-normalized and centered in a fixed-size image.
    """
    # Initialize logger
    logger = LOGGER.getChild('mnist-dataset')

    def __init__(self, path: str, batch_size: int):
        """Initialize MNIST dataset object.

        Args:
            path (str): Dataset path
            batch_size (int): Dataset batch size
        """
        # Create transform for loaders
        self.logger.info("Initializing transform.")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Verify train data
        self.logger.info("Verifying/downloading train data.")
        train_data = datasets.MNIST(
            root =          path,
            download =      True,
            train =         True,
            transform =     transform
        )

        # Verify test data
        self.logger.info("Verifying/downloading test data.")
        test_data = datasets.MNIST(
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
        self.channels_in =   1
        self.dim =          16

    def get_loaders(self) -> typing.Tuple[data.DataLoader, data.DataLoader]:
        """Fetch data loaders.

        Returns:
            typing.Tuple[data.DataLoader, data.DataLoader]: MNIST train loader, test loader
        """
        return self.train_loader, self.test_loader
    
    def __str__(self) -> str:
        """Provide string format of class.

        Returns:
            str: String format of MNIST dataset
        """
        return f"MNIST dataset ({self.num_classes} classes)"
    
if __name__ == '__main__':
    """Test dataset."""

    # Initialize dataset
    dataset = MNIST('./data', 64)

    # Print dataset __str__
    print(f"{colored('DATASET', 'magenta')}: \n{dataset}")

    # Print train loader
    print(f"\n{colored('TRAIN LOADER', 'magenta')}: \n{vars(dataset.train_loader)}")

    # Print test loader
    print(f"\n{colored('TEST LOADER', 'magenta')}: \n{vars(dataset.test_loader)}")
