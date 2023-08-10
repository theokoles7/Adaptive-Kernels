__all__ = ['caltech', 'cifar10', 'cifar100', 'imagenet', 'mnist', 'svhn']

from datasets.accessor import get_dataset

from datasets.caltech import CalTech
from datasets.cifar10 import Cifar10
from datasets.cifar100 import Cifar100
from datasets.imagenet import ImageNet
from datasets.mnist import MNIST
from datasets.svhn import SVHN