__all__ = ["__base__", "cifar10", "cifar100", "imagenet", "mnist"]

from datasets.__base__  import Dataset

from datasets.cifar10   import Cifar10
from datasets.cifar100  import Cifar100
from datasets.imagenet  import ImageNet
from datasets.mnist     import MNIST

def load_dataset(
    dataset:    str,
    **kwargs
) -> Dataset:
    """# Load specified dataset.

    ## Args:
        * dataset   (str):  Dataset selection.

    ## Returns:
        * Dataset:  Selected dataset, initialized and ready for training.
    """
    # Match dataset selection
    match dataset:
        
        # Cifar-10
        case "cifar10":     return Cifar10(**kwargs)
        
        # Cifar-100
        case "cifar100":    return Cifar100(**kwargs)
        
        # ImageNet
        case "imagenet":    return ImageNet(**kwargs)
        
        # MNIST
        case "mnist":       return MNIST(**kwargs)
        
        # Invalid selection
        case _:             raise NotImplementedError(f"Invalid dataset selection: {dataset}")