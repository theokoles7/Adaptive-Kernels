"""MNIST dataset and utilities."""

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
        super(MNIST, self).__init__()
        
        # Initialize logger
        self.__logger__:        Logger =        LOGGER.getChild("mnist-dataset")
        
        # Create transform
        transform:              Compose =       Compose([
                                                    # Convert images to PyTorch tensors
                                                    ToTensor(),
                                                    
                                                    # Normalize pixel values
                                                    Normalize((0.5,), (0.5,))
                                                ])

        # Download/verify train data
        train_data:             MNIST =         MNIST(
                                                    root =      data_path,
                                                    download =  True,
                                                    train =     True,
                                                    transform = transform
                                                )

        # Download/verify test data
        test_data:              MNIST =         MNIST(
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
        self._num_classes_:     int =           10
        self._channels_in_:     int =           1
        self._dim_:             int =           16

        # Log for debugging
        self._logger.debug(f"DATASET: {self} | CLASSES: {self._num_classes_} | CHANNELS: {self._channels_in_} | DIM: {self._dim_}")
        self._logger.debug(f"TRAIN LOADER:\n{vars(self._train_loader_)}")
        self._logger.debug(f"TEST LOADER:\n{vars(self._test_loader_)}")
    
    def __str__(self) -> str:
        """# Provide string format of MNIST dataset object.

        ## Returns:
            * str:  String format of MNIST dataset
        """
        return f"MNIST dataset ({self._num_classes_} classes)"