"""Poisson filter."""

import math, random

from scipy import special
from termcolor import cprint
import torch
import torch.nn as nn

from utils.arguments import get_args
from kernel_configs import kernel_config

class PoissonFilter(nn.Conv2d):

    def __init__(self, rate: int | float = 0, kernel_size: int = 3, channels: int = 3):
        """Create an (x, y) coordinate grid of shape (kernel_size, kernel_size, 2).

        Args:
            rate (int, optional): Distribution rate (lambda). Defaults to 0.
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

        # Verify integrity of rate and beta data types
        if math.isnan(rate):      raise ValueError(f"Rate expected to be a number; Got {type(rate)}")

        # Fetch corresponding filter list for kernel type provided
        filter_list = kernel_config[args.kernel_type]
        print(f"KERNEL TYPE {args.kernel_type}: {filter_list}")

        # Randomly select primary filter from list
        filter_name = filter_list[random.randint(0, len(filter_list) - 1)]

        # Set rate & beta to 1 if kernel is static (center)
        if filter_name == 'static': rate = beta = 1
        print(f"LAMBDA (RATE): {rate}")

        # Create tensors of variables
        seeds = torch.arange(kernel_size)
        x_grid = seeds.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        # Calculate 2D Laplace Kernel:
        # This will be the PRODUCT of two Laplace distributions based
        # on variables defind in x_grid and y_grid
        poisson_kernel = ((rate**torch.sum(xy_grid, dim=-1)) / (special.factorial(torch.sum(xy_grid, dim=-1)) * (math.e**(rate))))

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
            
        print(f"{filter_name.upper()}:\n{poisson_kernel}")

        # Reshape kernel
        poisson_kernel = poisson_kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)

        # Insert kernel(s) and "turn off" back propagation
        self.weight.data = poisson_kernel
        self.weight.requires_grad = False

if __name__ == '__main__':
    LaplaceFilter()