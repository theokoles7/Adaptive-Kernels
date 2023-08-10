"""Poisson distribution utilities."""

import math, random

from scipy import special
import torch, torch.nn as nn

from utils.globals import *
from kernels.config import kernel_config

class PoissonKernel(nn.Conv2d):

    def __init__(self, channels: int = 3):
        """Create an (x, y) coordinate grid of shape (ARGS.kernel_size, ARGS.kernel_size, 2).

        Args:
            channels (int, optional): Input channels. Defaults to 3.
        """
        # Initialize Conv2d object
        super().__init__(
            in_channels=channels, 
            out_channels=channels, 
            kernel_size=ARGS.kernel_size, 
            groups=channels, 
            bias=False, 
            padding=(1 if ARGS.kernel_size == 3 else (2 if ARGS.kernel_size == 5 else 0))
        )

        # Fetch corresponding filter list for kernel type provided
        filter_list = kernel_config[ARGS.kernel_type]
        if ARGS.debug: LOGGER.debug(f"KERNEL TYPE {ARGS.kernel_type}: {filter_list}")

        # Randomly select primary filter from list
        filter_name = filter_list[random.randint(0, len(filter_list) - 1)]

        # Set ARGS.rate & beta to 1 if kernel is static (center)
        if filter_name == 'static': ARGS.rate = beta = 1

        # Create tensors of variables
        seeds = torch.arange(ARGS.kernel_size)
        x_grid = seeds.repeat(ARGS.kernel_size).view(ARGS.kernel_size, ARGS.kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        # Calculate 2D Laplace Kernel:
        # This will be the PRODUCT of two Laplace distributions based
        # on variables defind in x_grid and y_grid
        poisson_kernel = ((ARGS.rate**torch.sum(xy_grid, dim=-1)) / (special.factorial(torch.sum(xy_grid, dim=-1)) * (math.e**(ARGS.rate))))

        # Ensure sum of distribution is 1
        poisson_kernel /= torch.sum(poisson_kernel)

        if filter_name in ['top-right', 'bottom-left', 'bottom-right']:
            if filter_name == 'bottom-right':
                temp = poisson_kernel[0].clone()
                poisson_kernel[0] = poisson_kernel[2]
                poisson_kernel[2] = temp

            poisson_kernel = torch.index_select(
                poisson_kernel, 
                (0 if filter_name == 'bottom-left' else 1), 
                torch.LongTensor([2, 1, 0]))
            
        LOGGER.info(f"LAMBDA (RATE): {ARGS.rate}")
        LOGGER.info(f"{filter_name.upper()}:\n{poisson_kernel}")

        # Reshape kernel
        poisson_kernel = poisson_kernel.view(1, 1, ARGS.kernel_size, ARGS.kernel_size).repeat(channels, 1, 1, 1)

        # Insert kernel(s) and "turn off" back propagation
        self.weight.data = poisson_kernel
        self.weight.requires_grad = False

if __name__ == '__main__':
    PoissonKernel()