"""Cifar10 dataset utilities."""

import typing

import torch.utils.data as data
from torchvision import transforms, datasets

from utils.globals import *

class Cifar10():
    """The CIFAR-10 dataset (https://www.cs.toronto.edu/~kriz/cifar.html) consists 
    of 60000 32x32 colour images in 10 classes, with 6000 images 
    per class. There are 50000 training images and 10000 test images.
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
        
        # Create training loader
        train_data = datasets.CIFAR10(
            root=ARGS.dataset_path,
            download=True,
            train=True,
            transform=transform
        )

        # Create testing loader
        test_data = datasets.CIFAR10(
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
        self.num_classes =  10
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
            str: String format of Cifar10 dataset.
        """
        return f"Cifar10 dataset ({self.num_classes} classes)"