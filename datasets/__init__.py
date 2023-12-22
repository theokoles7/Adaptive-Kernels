__all__ = ['accessor', 'cifar10', 'cifar100', 'imagenet', 'mnist']

from datasets.cifar10   import Cifar10
from datasets.cifar100  import Cifar100
from datasets.imagenet  import ImageNet
from datasets.mnist     import MNIST

from datasets.accessor  import get_dataset