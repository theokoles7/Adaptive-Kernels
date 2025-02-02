"""Base implementation of dataset classes."""

from torch.utils.data   import DataLoader

class Dataset():
    """Base Dataset class."""
    
    def __init__(self, **kwargs):
        """# Initialize Dataset object."""
        
        raise NotImplementedError(f"Subclass must override method.")
    
    def get_loaders(self, **kwargs) -> tuple[DataLoader, DataLoader]:
        """# Fetch dataset loaders.

        ## Returns:
            * tuple[DataLoader, DataLoader]:
                * DataLoader for train set.
                * DataLoader for test set.
        """
        return self._train_loader_, self._test_loader_