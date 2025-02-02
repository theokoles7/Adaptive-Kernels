"""ImageNet dataset and utilities."""

from logging                import Logger
from typing                 import override

from torch.utils.data       import DataLoader
from torchvision.datasets   import ImageFolder
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from datasets.__base__      import Dataset
from utils                  import LOGGER

class ImageNet(Dataset):
    """ImageNet is an image database organized according to the WordNet hierarchy (currently only 
    the nouns), in which each node of the hierarchy is depicted by hundreds and thousands of images. 
    The project has been instrumental in advancing computer vision and deep learning research. The 
    data is available for free to researchers for non-commercial use.
    """

    @override
    def __init__(self, 
        batch_size: int,
        data_path:  str =   "data",
        **kwargs
    ):
        """# Initialize ImageNet dataset loaders.

        ## Args:
            * batch_size    (int):              Dataset batch size.
            * data_path     (str, optional):    Path at which dataset is located/can be downloaded. 
                                                Defaults to "./data/".
        """
        # Initialize parent class
        super(ImageNet, self).__init__()
        
        # Initialize logger
        self.__logger__:        Logger =        LOGGER.getChild("imagenet-dataset")
        
        # Create transform
        transform:              Compose =       Compose([
                                                    # Resize images to 32x32 pixels
                                                    Resize(32),
                                                    
                                                    # Convert images to PyTorch tensors
                                                    ToTensor(),
                                                    
                                                    # Normalize pixel values
                                                    Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))
                                                ])

        # Verify train data
        train_data:             ImageFolder =   ImageFolder(
                                                    root =      f"{data_path}/tiny-imagenet-200/train",
                                                    transform = transform
                                                )

        # Verify test data
        test_data:              ImageFolder =   ImageFolder(
                                                    root =      f"{data_path}/tiny-imagenet-200/val",
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
        self._num_classes_:     int =           200
        self._channels_in_:     int =           3
        self._dim_:             int =           32

        # Log for debugging
        self._logger.debug(f"DATASET: {self} | CLASSES: {self._num_classes_} | CHANNELS: {self._channels_in_} | DIM: {self._dim_}")
        self._logger.debug(f"TRAIN LOADER:\n{vars(self._train_loader_)}")
        self._logger.debug(f"TEST LOADER:\n{vars(self._test_loader_)}")
    
    def __str__(self) -> str:
        """# Provide string format of ImageNet dataset object.

        ## Returns:
            * str:  String format of ImageNet dataset
        """
        return f"ImageNet dataset ({self._num_classes_} classes)"