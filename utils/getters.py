"""Dataset utilities."""

from datasets import *
from utils.globals import *
    
def get_args_banner() -> None:
    """Log important arguments for the record.
    """
    return (
        f"DATASET: {ARGS.dataset} BATCH_SIZE: {ARGS.batch_size} | "
        f"MODEL: {ARGS.model} LR: {ARGS.learning_rate} | "
        f"DISTRO: {ARGS.distribution} {get_distro_params(ARGS.distribution)}"
    )

def get_distro_params(distro: str) -> str:
    """Provide distribution parameters.

    Args:
        distro (str): Distribution argument

    Returns:
        str: Distribution parameters
    """
    if distro == 'poisson': return f"RATE: {ARGS.rate}"
    return f"LOC: {ARGS.location} SCALE: {ARGS.scale}"

def get_dataset():
    """Fetch selected dataset."""
    match ARGS.dataset:
        case 'caltech':     return caltech.CalTech()
        case 'cifar10':     return cifar10.Cifar10()
        case 'cifar100':    return cifar100.Cifar100()
        case 'imagenet':    return imagenet.ImageNet()
        case 'mnist':       return mnist.MNIST()
        case 'svhn':        return svhn.SVHN()

        case _:
            raise NotImplementedError(f"{ARGS.dataset} not supported.")

# def get_model() -> nn.Module:
#     """Fetch model.

#     Returns:
#         nn.Module: Selected model
#     """
#     match ARGS.model:
#         case 'normal':
#         case 'resnet':
#         case 'vgg':
#         case _:
#             raise NotImplementedError(f"{ARGS.model} not supported")