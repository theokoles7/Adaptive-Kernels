"""Cifar100 dataset utilities."""

import typing

import torch.utils.data as data
from torchvision import transforms, datasets

from utils.globals import *

class Cifar100():
    """This dataset is just like the CIFAR-10 
    (https://www.cs.toronto.edu/~kriz/cifar.html), except it has 100 
    classes containing 600 images each. There are 500 training images 
    and 100 testing images per class. The 100 classes in the CIFAR-100 
    are grouped into 20 superclasses. Each image comes with a "fine" 
    label (the class to which it belongs) and a "coarse" label (the 
    superclass to which it belongs). 
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
        train_data = datasets.CIFAR100(
            root=ARGS.dataset_path,
            download=True,
            train=True,
            transform=transform
        )
        
        # Create testing loader
        test_data = datasets.CIFAR100(
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

        # Record number of classes in dataset and number of input channels
        self.num_classes = 100
        self.channels_in =   3

    def get_loaders(self) -> typing.Tuple[data.DataLoader, data.DataLoader]:
        """Fetch data loaders.

        Returns:
            typing.Tuple[data.DataLoader, data.DataLoader]: Train loader, test loader
        """
        return self.train_loader, self.test_loader