__all__ = ['accessor', 'cifar10', 'cifar100', 'mnist', 'svhn']

from datasets.accessor  import get_dataset

from datasets.cifar10   import Cifar10
from datasets.cifar100  import Cifar100
from datasets.mnist     import MNIST
from datasets.svhn      import SVHN