"""Laplace filter."""

import math, random

from termcolor import cprint
import torch
import torch.nn as nn

from utils.arguments import get_args
from kernel_configs import kernel_config

class LaplaceFilter(nn.Conv2d):

    def __init__(self, location: int | float = 0, scale: int | float = 1, kernel_size: int = 3, channels: int = 3):
        """Create an (x, y) coordinate grid of shape (kernel_size, kernel_size, 2).

        Args:
            location (int, optional): Distribution location (mu). Defaults to 0.
            scale (int, optional): Distribution scale (beta). Defaults to 1.
            kernel_size (int, optional): Kernel dimension. Defaults to 3.
            channels (int, optional): Input channels. Defaults to 3.
        """
        # Initialize Conv2d object
        super().__init__(
            in_channels=channels, 
            out_channels=channels, 
            kernel_size=kernel_size, 
            groups=channels, 
            bias=False, 
            padding=(1 if kernel_size == 3 else (2 if kernel_size == 5 else 0))
        )

        # Parse arguments
        args = get_args()

        # Verify integrity of location and scale data types
        if math.isnan(location):      raise ValueError(f"Location expected to be a number; Got {type(location)}")
        if math.isnan(scale):   raise ValueError(f"Scale expected to be a number; Got {type(scale)}")

        # Fetch corresponding filter list for kernel type provided
        filter_list = kernel_config[args.kernel_type]
        print(f"KERNEL TYPE {args.kernel_type}: {filter_list}")

        # Randomly select primary filter from list
        filter_name = filter_list[random.randint(0, len(filter_list) - 1)]

        # Set location & scale to 1 if kernel is static (center)
        if filter_name == 'static': location = scale = 1
        print(f"MU (LOCATION): {location} | BETA (SCALE): {scale}")

        # Create tensors of variables
        seeds = torch.arange(kernel_size)
        x_grid = seeds.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        # Calculate 2D Laplace Kernel:
        # This will be the PRODUCT of two Laplace distributions based
        # on variables defind in x_grid and y_grid

        laplace_kernel = (1. / (2. * scale)) * (
            torch.exp(
                -abs(torch.sum(xy_grid, dim=-1) - location) / scale
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
            
        print(f"{filter_name.upper()}:\n{laplace_kernel}")

        # Reshape kernel
        laplace_kernel = laplace_kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)

        # Insert kernel(s) and "turn off" back propagation
        self.weight.data = laplace_kernel
        self.weight.requires_grad = False

if __name__ == '__main__':
    LaplaceFilter()