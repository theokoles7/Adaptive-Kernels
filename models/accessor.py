"""Model accessor."""

import torch.nn as nn

from utils.globals import ARGS

from models.normal import NormalCNN
from models.resnet import Resnet
from models.vgg import VGG

def get_model(channels_in: int, channels_out: int) -> nn.Module:
    """Fetch model.

    Args:
        channels_out (int): Input channels (Likely 1 if images are B&W, 3 if colored)
        channels_out (int): Output channels (number of classes in dataset)

    Returns:
        nn.Module: Selected model
    """
    match ARGS.model:
        case 'normal':  return NormalCNN(channels_in, channels_out)
        case 'resnet':  return Resnet(channels_in, channels_out)
        case 'vgg':     return VGG(channels_in, channels_out)
            
        case _: raise NotImplementedError(f"{ARGS.model} not supported")