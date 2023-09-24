"""SVHN utilities."""

import typing

from termcolor import colored
import torch.utils.data as data
from torchvision import transforms, datasets

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

    def __init__(self, path: str, batch_size: int):
        """Initialize SVHN dataset object.

        Args:
            path (str): Dataset path
            batch_size (int): Dataset batch size
        """
        # Create transform for loaders
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))
        ])

        # Verify training data
        train_data = datasets.SVHN(
            root =          path,
            download =      True,
            split =         'train',
            transform =     transform
        )

        # Verify testing data
        test_data = datasets.SVHN(
            root =          path,
            download =      True,
            split =         'test',
            transform =     transform
        )

        # Create training loader
        self.train_loader = data.DataLoader(
            train_data,
            batch_size =    batch_size,
            pin_memory =    True,
            num_workers =   4,
            shuffle =       True,
            drop_last =     True
        )

        # Create testing loader
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

    def get_loaders(self) -> typing.Tuple[data.DataLoader, data.DataLoader]:
        """Fetch data loaders.

        Returns:
            typing.Tuple[data.DataLoader, data.DataLoader]: Train loader, test loader
        """
        return self.train_loader, self.test_loader
    
    def __str__(self) -> str:
        """Provide str format of class.

        Returns:
            str: String format of SVHN dataset.
        """
        return f"SVHN dataset ({self.num_classes} classes)"
    
if __name__ == '__main__':
    """Test dataset."""

    # Initialize dataset
    dataset = SVHN('./data', 64)

    # Print dataset __str__
    print(f"{colored('DATASET', 'magenta')}: \n{dataset}")

    # Print train loader
    print(f"\n{colored('TRAIN LOADER', 'magenta')}: \n{vars(dataset.train_loader)}")

    # Print test loader
    print(f"\n{colored('TEST LOADER', 'magenta')}: \n{vars(dataset.test_loader)}")