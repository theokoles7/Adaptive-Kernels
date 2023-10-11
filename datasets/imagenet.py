"""ImageNet utilities."""

import typing

from termcolor import colored
import torch.utils.data as data
from torchvision import transforms, datasets

from utils import ARGS, LOGGER

class ImageNet():
    """ImageNet is an image database organized according to the WordNet hierarchy 
    (currently only the nouns), in which each node of the hierarchy is depicted 
    by hundreds and thousands of images. The project has been instrumental in 
    advancing computer vision and deep learning research. The data is 
    available for free to researchers for non-commercial use.
    """
    # Initialize logger
    logger = LOGGER.getChild('imagenet-dataset')

    def __init__(self, path: str, batch_size: int):
        """Initialize ImageNet dataset object.

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
        train_data = datasets.ImageNet(
            root =          path,
            download =      True,
            split =         'train',
            transform =     transform
        )

        # Verify test data
        self.logger.info("Verifying/downloading test data.")
        test_data = datasets.ImageNet(
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
        self.num_classes = 200
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
            typing.Tuple[data.DataLoader, data.DataLoader]: ImageNet train loader, test loader
        """
        return self.train_loader, self.test_loader
    
    def __str__(self) -> str:
        """Provide string format of class.

        Returns:
            str: String format of ImageNet dataset
        """
        return f"Cifar10 dataset ({self.num_classes} classes)"
    
if __name__ == '__main__':
    """Test dataset."""

    # Initialize dataset
    dataset = ImageNet('./data', 64)

    # Print dataset __str__
    print(f"\n{colored('DATASET', 'magenta')}: \n{dataset}")

    # Print train loader
    print(f"\n{colored('TRAIN LOADER', 'magenta')}: \n{vars(dataset.train_loader)}")

    # Print test loader
    print(f"\n{colored('TEST LOADER', 'magenta')}: \n{vars(dataset.test_loader)}")