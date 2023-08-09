"""MNIST dataset utilities."""

import typing

import torch.utils.data as data
from torchvision import transforms, datasets

from utils.globals import *

class MNIST():
    """The MNIST (http://yann.lecun.com/exdb/mnist/) database of 
    handwritten digits, available from this page, has a training set of 
    60,000 examples, and a test set of 10,000 examples. It is a 
    subset of a larger set available from NIST. The digits have been 
    size-normalized and centered in a fixed-size image.
    """

    def __init__(self):
        """Initialize train & test loaders for CalTech dataset.
        """
        # Create transform for loaders
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,))
            ]
        )
        
        # Create training loader
        train_data = datasets.MNIST(
            root=ARGS.dataset_path,
            download=True,
            train=True,
            transform=transform
        )

        # Create testing loader
        test_data = datasets.MNIST(
            root=ARGS.dataset_path,
            download=True,
            train=False,
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
        self.channels_in =   1

    def get_loaders(self) -> typing.Tuple[data.DataLoader, data.DataLoader]:
        """Fetch data loaders.

        Returns:
            typing.Tuple[data.DataLoader, data.DataLoader]: Train loader, test loader
        """
        return self.train_loader, self.test_loader
    
    def __str__(self) -> str:
        """Provide str format of class.

        Returns:
            str: String format of MNIST dataset.
        """
        return f"MNIST dataset ({self.num_classes} classes)"