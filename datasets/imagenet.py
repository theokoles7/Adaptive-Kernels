"""ImageNet dataset and utilities."""

import os, torch.utils.data as data, typing
from torchvision    import datasets, transforms

from utils          import ARGS, LOGGER

class ImageNet():
    """ImageNet is an image database organized according to the WordNet hierarchy 
    (currently only the nouns), in which each node of the hierarchy is depicted 
    by hundreds and thousands of images. The project has been instrumental in 
    advancing computer vision and deep learning research. The data is 
    available for free to researchers for non-commercial use.
    """

    # Initialize logger
    _logger = LOGGER.getChild('imagenet-dataset')

    def __init__(self, path: str = ARGS.dataset_path, batch_size: int = ARGS.batch_size):
        """Initialize ImageNet dataset loaders.

        Args:
            path (str, optional): Path at which dataset is located/can be downloaded. Defaults to ARGS.dataset_path.
            batch_size (int, optional): Dataset batch size. Defaults to ARGS.batch_size.
        """
        # Create transform for loaders
        self._logger.info("Initializing transform.")
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))
        ])

        # Verify train data
        self._logger.info("Verifying/downloading train data.")
        train_data = datasets.ImageFolder(
            root =          os.path.join(path, 'tiny-imagenet-200', 'train'),
            transform =     transform
        )

        # Verify test data
        self._logger.info("Verifying/downloading test data.")
        test_data = datasets.ImageFolder(
            root =          os.path.join(path, 'tiny-imagenet-200', 'val'),
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
        self.num_classes = 200
        self.channels_in =   3
        self.dim =          32

        self._logger.debug(f"DATASET: {self} | CLASSES: {self.num_classes} | CHANNELS: {self.channels_in} | DIM: {self.dim}")
        self._logger.debug(f"TRAIN LOADER:\n{vars(self.train_loader)}")
        self._logger.debug(f"TEST LOADER:\n{vars(self.test_loader)}")

    def get_loaders(self) -> typing.Tuple[data.DataLoader, data.DataLoader]:
        """Fetch data loaders.

        Returns:
            typing.Tuple[data.DataLoader, data.DataLoader]: ImageNet train & test loaders
        """
        return self.train_loader, self.test_loader
    
    def __str__(self) -> str:
        """Provide string format of ImageNet dataset object.

        Returns:
            str: String format of ImageNet dataset
        """
        return f"ImageNet dataset ({self.num_classes} classes)"