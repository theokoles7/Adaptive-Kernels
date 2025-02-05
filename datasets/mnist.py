"""MNIST dataset and utilities."""

from json                   import dumps
from logging                import Logger
from typing                 import override

from torch.utils.data       import DataLoader
from torchvision.datasets   import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor

from datasets.__base__      import Dataset
from utils                  import LOGGER

class MNIST(Dataset):
    """The MNIST (http://yann.lecun.com/exdb/mnist/) database of handwritten digits, available from 
    this page, has a training set of 60,000 examples, and a test set of 10,000 examples. It is a 
    subset of a larger set available from NIST. The digits have been size-normalized and centered 
    in a fixed-size image.
    """

    @override
    def __init__(self, 
        batch_size: int,
        data_path:  str =   "data",
        **kwargs
    ):
        """# Initialize MNIST dataset loaders.

        ## Args:
            * batch_size    (int):              Dataset batch size.
            * data_path     (str, optional):    Path at which dataset is located/can be downloaded. 
                                                Defaults to "./data/".
        """
        # Initialize parent class
        super(MNIST, self)
        
        # Initialize logger
        self.__logger__:        Logger =        LOGGER.getChild(suffix = "mnist-dataset")
        
        # Create transform
        transform:              Compose =       Compose([
                                                    # Convert images to PyTorch tensors
                                                    ToTensor(),
                                                    
                                                    # Normalize pixel values
                                                    Normalize((0.5,), (0.5,))
                                                ])
        self.__logger__.debug(f"Initialized data transform:\n{dumps(vars(transform), indent = 2, default = str)}")

        # Download/verify train data
        train_data:             MNIST =         MNIST(
                                                    root =      data_path,
                                                    download =  True,
                                                    train =     True,
                                                    transform = transform
                                                )
        self.__logger__.debug(f"Initialized training data:\n{dumps(vars(train_data), indent = 2, default = str)}")

        # Download/verify test data
        test_data:              MNIST =         MNIST(
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
        self._classes_:     int =           10
        self._channels_:    int =           1
        self._dimension_:   int =           16
        self.__logger__.debug(f"Attributes defined: _num_classes_ = {self._classes_}, _channels_in_ = {self._channels_}, _dim_ = {self._dimension_}")
    
    def __str__(self) -> str:
        """# Provide string format of MNIST dataset object.

        ## Returns:
            * str:  String format of MNIST dataset
        """
        return f"MNIST dataset ({self._classes_} classes)"