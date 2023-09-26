"""Gaussian distribution utilities."""

import math, random

import torch, torch.nn as nn

from kernels.config import kernel_config
from utils import ARGS, LOGGER

class GaussianKernel(nn.Conv2d):
    """Gaussian kernel class."""

    logger = LOGGER.getChild('gaussian-kernel')

    def __init__(self, location: float = 0, scale: float = 1, channels: int = 3, size: int = 3):
        """Create an (x, y) coordinate grid, using Gaussian distribution of convoluted data.

        Args:
            location (float, optional): Location parameter (mu) of Gaussian distribution. Defaults to 0.
            scale (float, optional): Scale parameter (sigma) of Gaussian distribution. Defaults to 1.
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

        # Calculate 2D Gaussian Kernel:
        # This will be the PRODUCT of two Gaussian distributions
        # based on vairables defined in x_grid and y_grid
        gaussian_kernel = (
            (1. / (2. * math.pi * (scale)**2)) * torch.exp(
                -torch.sum(
                    (xy_grid - location)**2., dim=-1
                ) /\
                (2 * (scale)**2)
            )
        )

        # Ensure sum of distribution is 1
        gaussian_kernel /= torch.sum(gaussian_kernel)

        if filter_name in ['top-right', 'bottom-left', 'bottom-right']:

            if filter_name == 'bottom-right':
                # Swap first and last kernels
                temp =                  gaussian_kernel[0].clone()
                gaussian_kernel[0] =    gaussian_kernel[2]
                gaussian_kernel[2] =    temp

            gaussian_kernel = torch.index_select(
                gaussian_kernel,
                (0 if filter_name == 'bottom-left' else 1),
                torch.LongTensor([2, 1, 0])
            )

        self.logger.info(f"MU (LOCATION): {location} | SIGMA (SCALE): {scale} | VAIRANCE: {scale**2}")
        self.logger.info(f"{filter_name.upper()}\n{gaussian_kernel}")

        # Reshape kernel
        gaussian_kernel = gaussian_kernel.view(1, 1, size, size).repeat(channels, 1, 1, 1)

        self.weight.data =          gaussian_kernel
        self.weight.requires_grad = False

if __name__ == '__main__':
    """Test Gaussian kernel."""

    GaussianKernel()