"""Cauchy distribution utilities."""

import math, random

import torch, torch.nn as nn

from kernels.config import kernel_config
from utils import ARGS, LOGGER

class CauchyKernel(nn.Conv2d):
    """Cauchy kernel class."""

    logger = LOGGER.getChild('cauchy-kernel')

    def __init__(self, location: float = 0, scale: float = 1, channels: int = 3, size: int = 3):
        """Create an (x, y) coordinate grid, using Cauchy distribution of convoluted data.

        Args:
            location (float, optional): Location parameter (chi) of Cauchy distribution. Defaults to 0.
            scale (float, optional): Scale parameter (gamma) of Cauchy distribution. Defaults to 1.
            channels (int, optional): Input channels. Defaults to 3.
            size (int, optional): Kernel size (square). Defaults to 3.
        """
        # Initialize Conv2d object
        super().__init__(
            in_channels =   channels,
            out_channels =  channels,
            kernel_size =   size,
            groups =        channels,
            bias =          False,
            padding =       (1 if size == 3 else (2 if size == 5 else 0))
        )

        # Fetch appropriate filter list
        filter_list = kernel_config[ARGS.kernel_type]
        if ARGS.debug: self.logger.debug(f"KERNEL TYPE {ARGS.kernel_type}: {filter_list}")

        # Randomly select primary filter from list
        filter_name = filter_list[random.randint(0, len(filter_list) - 1)]
        if ARGS.debug: self.logger.debug(f"FILTER NAME: {filter_name}")

        # Set location & scale to 1 if kernel is static (center)
        if filter_name == 'static': location = scale = 1

        # Create tensors
        seeds =     torch.arange(size)
        x_grid =    seeds.repeat(size).view(size, size)
        y_grid =    x_grid.t()
        xy_grid =   torch.stack([x_grid, y_grid], dim=-1).float()

        # Calculate 2D Cauchy Kernel:
        # This will be the PRODUCT of two Gaussian distributions
        # based on vairables defined in x_grid and y_grid
        cauchy_kernel = (
            1. / (
                (math.pi * scale) * (1. + (
                    (torch.sum(xy_grid, dim=-1) - location) / (scale)
                )**2)
            )
        )

        # If NAN or INF detected, raise error
        if torch.isinf(cauchy_kernel).any():
            raise ValueError(f"NAN or INF detected in kernel:\n{cauchy_kernel}")

        # Ensure sum of distribution is 1
        cauchy_kernel /= torch.sum(cauchy_kernel)

        if filter_name in ['top-right', 'bottom-left', 'bottom-right']:

            if filter_name == 'bottom-right':
                # Swap first and last kernels
                temp =                  cauchy_kernel[0].clone()
                cauchy_kernel[0] =    cauchy_kernel[2]
                cauchy_kernel[2] =    temp

            cauchy_kernel = torch.index_select(
                cauchy_kernel,
                (0 if filter_name == 'bottom-left' else 1),
                torch.LongTensor([2, 1, 0])
            )

        self.logger.info(f"CHI (LOCATION): {location} | GAMMA (SCALE): {scale} | VAIRANCE: {scale**2}")
        self.logger.info(f"{filter_name.upper()}\n{cauchy_kernel}")

        # Reshape kernel
        cauchy_kernel = cauchy_kernel.view(1, 1, size, size).repeat(channels, 1, 1, 1)

        self.weight.data =          cauchy_kernel
        self.weight.requires_grad = False

if __name__ == '__main__':
    """Test Gaussian kernel."""

    CauchyKernel()