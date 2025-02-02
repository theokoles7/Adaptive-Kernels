__all__ = ["__base__", "cifar10", "cifar100", "imagenet", "mnist"]

from datasets.__base__  import Dataset

from datasets.cifar10   import Cifar10
from datasets.cifar100  import Cifar100
from datasets.imagenet  import ImageNet
from datasets.mnist     import MNIST

def load_dataset(
    dataset:    str,
    batch_size: int =   64,
    data_path:  str =   "data",
    **kwargs
) -> Dataset:
    """# Load specified dataset.

    ## Args:
        * dataset       (str):              Dataset selection.
        * batch_size    (int, optional):    Dataset batch size. Defaults to 64.
        * data_path     (str, optional):    Path at which dataset will be downloaded/loaded. 
                                            Defaults to "./data/".

    ## Returns:
        * Dataset:  Selected dataset, initialized and ready for training.
    """
    # Match dataset selection
    match dataset:
        
        # Cifar-10
        case "cifar10":     return Cifar10(**locals())
        
        # Cifar-100
        case "cifar100":    return Cifar100(**locals())
        
        # ImageNet
        case "imagenet":    return ImageNet(**locals())
        
        # MNIST
        case "mnist":       return MNIST(**locals())
        
        # Invalid selection
        case _:             raise NotImplementedError(f"Invalid dataset selection: {dataset}")