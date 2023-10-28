"""Resnet model block."""

import torch, torch.nn as nn, torch.nn.functional as F

from kernels.accessor import get_kernel
from utils import ARGS, LOGGER

class ResnetBlock(nn.Module):
    """Block component for Resnet model."""
    expansion = 1

    # Initialize logger
    logger = LOGGER.getChild('resnet-block')

    def __init__(self, channels_in: int, channels_out: int, stride: int):
        """Initialize Resnet block.

        Args:
            channels_in (int): Input channels
            channels_out (int): Output channels
            stride (int): Kernel convolution stride
        """
        super(ResnetBlock, self).__init__()

        # Initialize distribution parameters
        self.location =     [ARGS.location]*2
        self.scale =        [ARGS.scale]*2
        self.rate =         [ARGS.rate]*2

        self.channels_out = channels_out

        # Batch normalization layers
        self.bn1 =          nn.BatchNorm2d(channels_out)
        self.bn2 =          nn.BatchNorm2d(channels_out)

        # Convolving layers
        self.conv1 =        nn.Conv2d(channels_in,  channels_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 =        nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1,      padding=1, bias=False)

        self.shortcut =     nn.Sequential()
        # if stride != 1 or channels_in != (channels_out * self.expansion):
        #     self.shortcut_kernel = True
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(channels_in, (channels_out * self.expansion), kernel_size=1, stride=stride, bias=False),
        #         nn.BatchNorm2d(channels_out * self.expansion)
        #     )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Feed input through network and return output.

        Args:
            X (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        if ARGS.debug: self.logger.debug(f"RESNET BLOCK X SHAPE (step 1): {X.shape}")
        X = self.conv1(X)

        if ARGS.debug: self.logger.debug(f"RESNET BLOCK X SHAPE (step 2): {X.shape}")
        with torch.no_grad():
            y = X.float()
            self.location[0] =  torch.mean(y).item()
            self.scale[0] =     torch.std(y).item()
            self.rate[0] =      torch.mean(y).item()

        if ARGS.debug: self.logger.debug(f"RESNET BLOCK X SHAPE (step 3): {X.shape}")
        X = self.conv2(F.relu(self.bn1(self.kernel1(X) if ARGS.distribution else X)))

        if ARGS.debug: self.logger.debug(f"RESNET BLOCK X SHAPE (step 4): {X.shape}")
        with torch.no_grad():
            y = X.float()
            self.location[1] =  torch.mean(y).item()
            self.scale[1] =     torch.std(y).item()
            self.rate[1] =      torch.mean(y).item()

        if ARGS.debug: self.logger.debug(f"RESNET BLOCK X SHAPE (step 5): {X.shape}")
        X = self.bn2(self.kernel2(X) if ARGS.distribution else X)

        if ARGS.debug: self.logger.debug(f"RESNET BLOCK X SHAPE (step 6): {X.shape}")

        X += self.shortcut(X)

        return F.relu(X)
    
    def update_kernels(self) -> None:
        """Update kernels."""
        self.kernel1 = get_kernel(
            ARGS.kernel_size, 
            self.channels_out,
            self.location[0],
            self.scale[0],
            self.rate[0]
        )

        self.kernel2 = get_kernel(
            ARGS.kernel_size, 
            self.channels_out,
            self.location[1],
            self.scale[1],
            self.rate[1]
        )