"""Dataset utilities."""

from termcolor import colored

from datasets.cifar10   import Cifar10
from datasets.cifar100  import Cifar100
from datasets.mnist     import MNIST
from datasets.svhn      import SVHN

def get_dataset(dataset: str, path: str, batch_size: int) -> Cifar10 | Cifar100 | MNIST | SVHN:
    """Provide appropriate dataset.

    Args:
        dataset (str): Dataset selection
        path (str): Dataset path
        batch_size (int): Dataset batch size

    Returns:
        Cifar10 | Cifar100 | MNIST | SVHN: Selected dataset
    """
    match dataset:
        case 'cifar10':     return Cifar10(  path, batch_size)
        case 'cifar100':    return Cifar100( path, batch_size)
        case 'mnist':       return MNIST(    path, batch_size)
        case 'svhn':        return SVHN(     path, batch_size)

        case _:
            raise NotImplementedError(
                f"{dataset} not supported."
            )
        
if __name__ == '__main__':
    "Test dataset accessor."

    # Access Cifar10 dataset
    print(f"\n{colored('Access Cifar10:', 'green')}: {get_dataset('cifar10', './data', 64)}")

    # Access Cifar100 dataset
    print(f"\n{colored('Access Cifar100:', 'green')}: {get_dataset('cifar100', './data', 64)}")

    # Access MNIST dataset
    print(f"\n{colored('Access MNIST:', 'green')}: {get_dataset('mnist', './data', 64)}")

    # Access SVHN dataset
    print(f"\n{colored('Access SVHN:', 'green')}: {get_dataset('svhn', './data', 64)}")

    # Access unsupported dataset
    print(colored("\nStack trace should follow this line, reporting that \'bogus_dataset\' is not supported.", 'red'))
    get_dataset('bogus_dataset', './data', 64)