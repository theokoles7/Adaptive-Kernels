"""Dataset accessor."""

from datasets   import Cifar10, Cifar100, ImageNet, MNIST

from utils  import ARGS, LOGGER

# Initialize logger
logger = LOGGER.getChild('dataset-accessor')

def get_dataset(dataset: str = ARGS.dataset, path: str = ARGS.dataset_path, batch_size: int = ARGS.batch_size) -> Cifar10 | Cifar100 | ImageNet | MNIST:
    """Provide appropriate dataset.

    Args:
        dataset (str, optional): Dataset selection. Defaults to ARGS.dataset.
        path (str, optional): Path at which dataset will be located/downloaded. Defaults to ARGS.dataset_path.
        batch_size (int, optional): Dataset batch size. Defaults to ARGS.batch_size.

    Returns:
        Cifar10 | Cifar100 | ImageNet | MNIST: Selected dataset
    """
    logger.info(f"Fetching dataset: {dataset}")

    match dataset:
        case 'cifar10':     return Cifar10(     path, batch_size)
        case 'cifar100':    return Cifar100(    path, batch_size)
        case 'imagenet':    return ImageNet(    path, batch_size)
        case 'mnist':       return MNIST(       path, batch_size)

        case _:
            raise NotImplementedError(
                f"{dataset} dataset not supported"
            )