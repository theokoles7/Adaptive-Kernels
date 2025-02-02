"""Cifar 100 dataset and utilities."""

from logging                import Logger
from typing                 import override

from torch.utils.data       import DataLoader
from torchvision.datasets   import CIFAR100
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from datasets.__base__      import Dataset
from utils                  import LOGGER

class Cifar100(Dataset):
    """This dataset is just like CIFAR-10 (https://www.cs.toronto.edu/~kriz/cifar.html), except it 
    has 100 classes containing 600 images each. There are 500 training images and 100 testing images 
    per class. The 100 classes in the CIFAR-100 are grouped into 20 superclasses. Each image comes 
    with a "fine" label (the class to which it belongs) and a "coarse" label (the superclass to 
    which it belongs). 
    """

    @override
    def __init__(self, 
        batch_size: int,
        data_path:  str =   "data",
        **kwargs
    ):
        """# Initialize Cifar100 dataset loaders.

        ## Args:
            * batch_size    (int):              Dataset batch size.
            * data_path     (str, optional):    Path at which dataset is located/can be downloaded. 
                                                Defaults to "./data/".
        """
        # Initialize parent class
        super(Cifar100, self).__init__()
        
        # Initialize logger
        self.__logger__:        Logger =        LOGGER.getChild("cifar100-dataset")
        
        # Create transform
        transform:              Compose =       Compose([
                                                    # Resize images to 32x32 pixels
                                                    Resize(32),
                                                    
                                                    # Convert images to PyTorch tensors
                                                    ToTensor(),
                                                    
                                                    # Normalize pixel values
                                                    Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))
                                                ])

        # Download/verify train data
        train_data:             CIFAR100 =      CIFAR100(
                                                    root =      data_path,
                                                    download =  True,
                                                    train =     True,
                                                    transform = transform
                                                )

        # Download/verify test data
        test_data:              CIFAR100 =      CIFAR100(
                                                    root =      data_path,
                                                    download =  True,
                                                    train =     False,
                                                    transform = transform
                                                )

        # Create training loader
        self._train_loader_:    DataLoader =    DataLoader(
                                                    dataset =       train_data,
                                                    batch_size =    batch_size,
                                                    pin_memory =    True,
                                                    num_workers =   4,
                                                    shuffle =       True,
                                                    drop_last =     True
                                                )

        # Create testing loader
        self._test_loader_:     DataLoader =    DataLoader(
                                                    dataset =       test_data,
                                                    batch_size =    batch_size,
                                                    pin_memory =    True,
                                                    num_workers =   4,
                                                    shuffle =       True,
                                                    drop_last =     False
                                                )

        # Define parameters (passed to model during initialization for layer dimensions)
        self._num_classes_:     int =           100
        self._channels_in_:     int =           3
        self._dim_:             int =           32

        # Log for debugging
        self._logger.debug(f"DATASET: {self} | CLASSES: {self._num_classes_} | CHANNELS: {self._channels_in_} | DIM: {self._dim_}")
        self._logger.debug(f"TRAIN LOADER:\n{vars(self._train_loader_)}")
        self._logger.debug(f"TEST LOADER:\n{vars(self._test_loader_)}")
    
    def __str__(self) -> str:
        """# Provide string format of Cifar100 dataset object.

        ## Returns:
            * str:  String format of Cifar100 dataset
        """
        return f"Cifar100 dataset ({self._num_classes_} classes)"