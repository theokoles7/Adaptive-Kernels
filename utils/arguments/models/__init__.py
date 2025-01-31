__all__ = ["normal", "resnet", "vgg"]

from utils.arguments.models.normal  import add_normal_cnn_parser
from utils.arguments.models.resnet  import add_resnet_parser
from utils.arguments.models.vgg     import add_vgg_parser