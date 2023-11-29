"""Gumbel distribution utilities."""

import random

import torch, torch.nn as nn

from kernels.config import kernel_config
from utils import ARGS, LOGGER

class GumbelKernel(nn.Conv2d):
    """Gumbel kernel class."""
    logger = LOGGER.getChild('gumbel-kernel')

    def __init__(self, location: float = 0, scale: float = 1, channels: int = 3, size: int = 3):
        """Create an (x, y) coordinate grid, using Gumbel distribution of convoluted data.

        Args:
            location (float, optional): Location parameter (mu) of Gumbel distribution. Defaults to 0.
            scale (float, optional): Scale parameter (beta) of Gumbel distribution. Defaults to 1.
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
            padding =       (1 if size == 3 else(2 if size == 5 else 0))
        )

        # Fetch corresponding filter list
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

        # Calculate 2D Gumbel Kernel:
        # This will be the PRODUCT of two Gumbel distributions
        # based on vairables defined in x_grid and y_grid
        gumbel_kernel = (
            (1 / scale) * torch.exp(
                -(
                    ((torch.sum(xy_grid - location, dim=-1)) / scale) +\
                        (torch.exp(((torch.sum(xy_grid - location, dim=-1)) / scale)))
                )
            )
        )

        # If NAN or INF detected, raise error
        if torch.isinf(gumbel_kernel).any():
            raise ValueError(f"NAN or INF detected in kernel:\n{gumbel_kernel}")

        # Ensure sum of distribution is 1
        gumbel_kernel /= torch.sum(gumbel_kernel)

        if filter_name in ['top-right', 'bottom-left', 'bottom-right']:

            if filter_name == 'bottom-right':
                # Swap first and last kernels
                temp =                gumbel_kernel[0].clone()
                gumbel_kernel[0] =    gumbel_kernel[2]
                gumbel_kernel[2] =    temp

            gumbel_kernel = torch.index_select(
                gumbel_kernel,
                (0 if filter_name == 'bottom-left' else 1),
                torch.LongTensor([2, 1, 0])
            )

        self.logger.info(f"MU (LOCATION): {location} | BETA (SCALE): {scale}")
        self.logger.info(f"{filter_name.upper()}\n{gumbel_kernel}")

        # Reshape kernel
        gumbel_kernel = gumbel_kernel.view(1, 1, size, size).repeat(channels, 1, 1, 1)

        self.weight.data =          gumbel_kernel
        self.weight.requires_grad = False

if __name__ == '__main__':
    """Test Gumbel kernel."""

    GumbelKernel()