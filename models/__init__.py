__all__ = ['accessor', 'normal_cnn', 'resnet', 'vgg']

from models.accessor import get_model

from models.normal_cnn      import NormalCNN
from models.resnet          import Resnet
from models.resnet_block    import ResnetBlock
from models.vgg             import VGG