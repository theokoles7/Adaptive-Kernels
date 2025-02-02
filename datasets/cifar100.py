"""Cifar 100 dataset and utilities."""

from json                   import dumps
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
        super(Cifar100, self)
        
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
        self.__logger__.debug(f"Initialized data transform:\n{dumps(vars(transform), indent = 2, default = str)}")

        # Download/verify train data
        train_data:             CIFAR100 =      CIFAR100(
                                                    root =      data_path,
                                                    download =  True,
                                                    train =     True,
                                                    transform = transform
                                                )
        self.__logger__.debug(f"Initialized training data:\n{dumps(vars(train_data), indent = 2, default = str)}")

        # Download/verify test data
        test_data:              CIFAR100 =      CIFAR100(
                                                    root =      data_path,
                                                    download =  True,
                                                    train =     False,
                                                    transform = transform
                                                )
        self.__logger__.debug(f"Initialized testing data:\n{dumps(vars(test_data), indent = 2, default = str)}")

        # Create training loader
        self._train_loader_:    DataLoader =    DataLoader(
                                                    dataset =       train_data,
                                                    batch_size =    batch_size,
                                                    pin_memory =    True,
                                                    num_workers =   4,
                                                    shuffle =       True,
                                                    drop_last =     True
                                                )
        self.__logger__.debug(f"Initialized training data loader:\n{dumps(vars(self._train_loader_), indent = 2, default = str)}")

        # Create testing loader
        self._test_loader_:     DataLoader =    DataLoader(
                                                    dataset =       test_data,
                                                    batch_size =    batch_size,
                                                    pin_memory =    True,
                                                    num_workers =   4,
                                                    shuffle =       True,
                                                    drop_last =     False
                                                )
        self.__logger__.debug(f"Initialized testing data loader:\n{dumps(vars(self._test_loader_), indent = 2, default = str)}")

        # Define parameters (passed to model during initialization for layer dimensions)
        self._num_classes_:     int =           100
        self._channels_in_:     int =           3
        self._dim_:             int =           32
        self.__logger__.debug(f"Attributes defined: _num_classes_ = {self._num_classes_}, _channels_in_ = {self._channels_in_}, _dim_ = {self._dim_}")
    
    def __str__(self) -> str:
        """# Provide string format of Cifar100 dataset object.

        ## Returns:
            * str:  String format of Cifar100 dataset
        """
        return f"Cifar100 dataset ({self._num_classes_} classes)"