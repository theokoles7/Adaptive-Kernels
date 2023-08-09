"""ImageNet dataset utilities."""

import typing

import torch.utils.data as data
from torchvision import transforms, datasets

from utils.globals import *

class ImageNet():
    """ImageNet (https://image-net.org/) is an image database organized 
    according to the WordNet hierarchy (currently only the nouns), in 
    which each node of the hierarchy is depicted by hundreds and thousands 
    of images. The project has been instrumental in advancing computer 
    vision and deep learning research. The data is available for free to 
    researchers for non-commercial use.
    """

    def __init__(self):
        """Initialize train & test loaders for CalTech dataset.
        """
        # Create transform for loaders
        transform = transforms.Compose(
            [transforms.Resize(32),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))
            ]
        )
        
        # Create transform for loaders
        train_data = datasets.ImageFolder(
            ARGS.dataset_path,
            transform=transform
        )
        
        # Create transform for loaders
        test_data = datasets.ImageFolder(
            ARGS.dataset_path,
            transform=transform
        )

        # Create training loader  
        self.train_loader = data.DataLoader(
            train_data, 
            batch_size=ARGS.batch_size,
            pin_memory=True,
            num_workers=4,
            shuffle=False,
            drop_last=True
        )

        # Create testing loader
        self.test_loader = data.DataLoader(
            test_data, 
            batch_size=ARGS.batch_size,
            pin_memory=True,
            num_workers=4,
            shuffle=True,
            drop_last=False,
        )

        # Dataset parameters
        self.num_classes = 200
        self.channels_in =   3

    def get_loaders(self) -> typing.Tuple[data.DataLoader, data.DataLoader]:
        """Fetch data loaders.

        Returns:
            typing.Tuple[data.DataLoader, data.DataLoader]: Train loader, test loader
        """
        return self.train_loader, self.test_loader
    
    def __str__(self) -> str:
        """Provide str format of class.

        Returns:
            str: String format of ImageNet dataset.
        """
        return f"ImageNet dataset ({self.num_classes} classes)"