"""Model utilities."""

from utils import ARGS, LOGGER

from models.normal_cnn  import NormalCNN
from models.resnet      import Resnet
from models.vgg         import VGG

# Initialize logger
logger = LOGGER.getChild('model-accessor')

def get_model(channels_in: int, channels_out: int, dim: int) -> NormalCNN | Resnet | VGG:
    """Fetch appropriate model.

    Args:
        channels_in (int): Input channels
        channels_out (int): Output channels
        dim (int): Input image dimension

    Returns:
        NormalCNN | Resnet | VGG: Selected model
    """
    match ARGS.model:
        case 'normal':  return NormalCNN(channels_in, channels_out, dim)
        case 'resnet':  return    Resnet(channels_in, channels_out)
        case 'vgg':     return       VGG(channels_in, channels_out)

        case _: raise NotImplementedError(f"{ARGS.model} not supported.")