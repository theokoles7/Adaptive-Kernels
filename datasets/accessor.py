"""Dataset accessor."""

from utils.globals import ARGS

from datasets.caltech import CalTech
from datasets.cifar10 import Cifar10
from datasets.cifar100 import Cifar100
from datasets.imagenet import ImageNet
from datasets.mnist import MNIST
from datasets.svhn import SVHN

def get_dataset():
    """Fetch selected dataset."""
    match ARGS.dataset:
        case 'caltech':     return CalTech()
        case 'cifar10':     return Cifar10()
        case 'cifar100':    return Cifar100()
        case 'imagenet':    return ImageNet()
        case 'mnist':       return MNIST()
        case 'svhn':        return SVHN()

        case _: raise NotImplementedError(f"{ARGS.dataset} not supported.")