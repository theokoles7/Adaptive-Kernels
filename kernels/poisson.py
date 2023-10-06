"""Poisson distribution utilities."""

import math, random

from scipy import special
import torch, torch.nn as nn

from kernels.config import kernel_config
from utils import ARGS, LOGGER

class PoissonKernel(nn.Conv2d):
    """Poisson kernel class."""

    logger = LOGGER.getChild('poisson-kernel')

    def __init__(self, rate: float = 1, channels: int = 3, size: int = 3):
        """Create an (x, y) coordinate grid, using Poisson distribution of convoluted data.

        Args:
            rate (float, optional): Rate parameter (lambda) of Poisson distribution. Defaults to 1.
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

        # Calculate 2D Poisson Kernel:
        # This will be the PRODUCT of two Poisson distributions
        # based on vairables defined in x_grid and y_grid
        poisson_kernel = ((rate**torch.sum(xy_grid, dim=-1)) * (math.e**(-rate))) / (special.factorial(torch.sum(xy_grid, dim=-1)))
        
        # (
        #     (rate**torch.sum(xy_grid, dim=-1)) / (
        #         (special.factorial(torch.sum(xy_grid, dim=-1)) * (math.e**(rate)))
        #     )
        # )

        # Ensure sum of distribution is 1
        poisson_kernel /= torch.sum(poisson_kernel)

        if filter_name in ['top-right', 'bottom-left', 'bottom-right']:

            if filter_name == 'bottom-right':
                # Swap first and last kernels
                temp =                  poisson_kernel[0].clone()
                poisson_kernel[0] =    poisson_kernel[2]
                poisson_kernel[2] =    temp

            poisson_kernel = torch.index_select(
                poisson_kernel,
                (0 if filter_name == 'bottom-left' else 1),
                torch.LongTensor([2, 1, 0])
            )

        self.logger.info(f"LAMBDA (RATE): {rate}")
        self.logger.info(f"{filter_name.upper()}\n{poisson_kernel}")

        # Reshape kernel
        poisson_kernel = poisson_kernel.view(1, 1, size, size).repeat(channels, 1, 1, 1)

        self.weight.data =          poisson_kernel
        self.weight.requires_grad = False

if __name__ == '__main__':
    """Test Gaussian kernel."""

    PoissonKernel()