"""Laplace distribution utilities."""

import math, random

import torch, torch.nn as nn

from utils.globals import *
from kernels.config import kernel_config

class LaplaceKernel(nn.Conv2d):

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

        # Set ARGS.location & ARGS.scale to 1 if kernel is static (center)
        if filter_name == 'static': ARGS.location = ARGS.scale = 1
        if ARGS.debug: LOGGER.debug(f"MU (LOCATION): {ARGS.location} | BETA (SCALE): {ARGS.scale}")

        # Create tensors of variables
        seeds = torch.arange(ARGS.kernel_size)
        x_grid = seeds.repeat(ARGS.kernel_size).view(ARGS.kernel_size, ARGS.kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        # Calculate 2D Laplace Kernel:
        # This will be the PRODUCT of two Laplace distributions based
        # on variables defind in x_grid and y_grid

        laplace_kernel = (1. / (2. * ARGS.scale)) * (
            torch.exp(
                -abs(torch.sum(xy_grid, dim=-1) - ARGS.location) / ARGS.scale
            )
        )

        # Ensure sum of distribution is 1
        laplace_kernel /= torch.sum(laplace_kernel)

        if filter_name in ['top-right', 'bottom-left', 'bottom-right']:
            if filter_name == 'bottom-right':
                temp = laplace_kernel[0].clone()
                laplace_kernel[0] = laplace_kernel[2]
                laplace_kernel[2] = temp

            laplace_kernel = torch.index_select(
                laplace_kernel, 
                (0 if filter_name == 'bottom-left' else 1), 
                torch.LongTensor([2, 1, 0]))
            
        LOGGER.info(f"{filter_name.upper()}:\n{laplace_kernel}")

        # Reshape kernel
        laplace_kernel = laplace_kernel.view(1, 1, ARGS.kernel_size, ARGS.kernel_size).repeat(channels, 1, 1, 1)

        # Insert kernel(s) and "turn off" back propagation
        self.weight.data = laplace_kernel
        self.weight.requires_grad = False

if __name__ == '__main__':
    LaplaceKernel()