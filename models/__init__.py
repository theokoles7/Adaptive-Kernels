__all__ = ['accessor', 'normal', 'resnet_block', 'resnet', 'vgg']

from models.normal          import NormalCNN
from models.resnet_block    import ResnetBlock
from models.resnet          import Resnet
from models.vgg             import VGG

from models.accessor        import get_model