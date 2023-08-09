"""Caltech dataset utilities."""

import typing

import torch.utils.data as data
from torchvision import transforms, datasets

from utils.globals import *

class CalTech():
    """(https://data.caltech.edu/records/mzrjq-6wc02) Pictures of objects belonging 
    to 101 categories. About 40 to 800 images per category. 
    Most categories have about 50 images. Collected in September 2003 by Fei-Fei Li, 
    Marco Andreetto, and Marc'Aurelio Ranzato. The size 
    of each image is roughly 300 x 200 pixels.
    """

    def __init__(self):
        """Initialize train & test loaders for CalTech dataset.
        """
        # Create transform for training loader
        train_transform = transforms.Compose(
            [transforms.CenterCrop(128),
             transforms.Resize(64),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))
            ]
        )

        # Create transform for testing loader
        test_transform = transforms.Compose(
            [transforms.CenterCrop(128),
             transforms.Resize(64),
             transforms.ToTensor(),
             transforms.Lambda(lambda x: x.repeat(3, 1, 1))
            ]
        )

        # Organize training data
        train_data = datasets.Caltech101(
            root=ARGS.dataset_path,
            download=True,
            transform=train_transform
        )

        # Organize testing data
        test_data = datasets.Caltech101(
            root=ARGS.dataset_path,
            download=True,
            transform=test_transform
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
        self.num_classes = 101
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
            str: String format of Caltech dataset.
        """
        return f"Caltech dataset ({self.num_classes} classes)"