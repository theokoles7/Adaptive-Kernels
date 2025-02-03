"""Base implementation of dataset classes."""

from torch.utils.data   import DataLoader

class Dataset():
    """Base Dataset class."""
    
    def __init__(self, **kwargs):
        """# Initialize Dataset object."""
        
        raise NotImplementedError(f"Subclass must override method.")
    
    def channels(self) -> int:
        """# Access dataset image channels.

        ## Returns:
            * int:  Number of channels in dataset images.
        """
        return self._channels_
    
    def classes(self) -> int:
        """# Access dataset classes.

        ## Returns:
            * int:  Number of classes in dataset.
        """
        return self._classes_
    
    def dimension(self) -> int:
        """# Access dataset image dimension.

        ## Returns:
            * int:  Dimension of dataset images.
        """
        return self._dimension_
    
    def get_loaders(self, **kwargs) -> tuple[DataLoader, DataLoader]:
        """# Fetch dataset loaders.

        ## Returns:
            * tuple[DataLoader, DataLoader]:
                * DataLoader for train set.
                * DataLoader for test set.
        """
        return self._train_loader_, self._test_loader_