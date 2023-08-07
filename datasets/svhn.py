"""SVHN dataset utilities."""

import typing

import torch.utils.data as data
from torchvision import transforms, datasets

from utils.globals import *

class SVHN():
    """SVHN (http://ufldl.stanford.edu/housenumbers/) is a real-world 
    image dataset for developing machine learning and object recognition 
    algorithms with minimal requirement on data preprocessing and formatting. 
    It can be seen as similar in flavor to MNIST (e.g., the images are of 
    small cropped digits), but incorporates an order of magnitude more 
    labeled data (over 600,000 digit images) and comes from a significantly 
    harder, unsolved, real world problem (recognizing digits and numbers 
    in natural scene images). SVHN is obtained from house numbers in 
    Google Street View images.
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
        train_data = datasets.SVHN(
            root=ARGS.dataset_path,
            download=True,
            split='train',
            transform=transform
        )

        # Create testing loader
        test_data = datasets.SVHN(
            root=ARGS.dataset_path,
            download=True,
            split='test',
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
        self.num_classes = 10
        self.channels_in =  3

    def get_loaders(self) -> typing.Tuple[data.DataLoader, data.DataLoader]:
        """Fetch data loaders.

        Returns:
            typing.Tuple[data.DataLoader, data.DataLoader]: Train loader, test loader
        """
        return self.train_loader, self.test_loader