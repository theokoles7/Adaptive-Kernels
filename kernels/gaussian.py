"""Gaussian distribution utilities."""

import math, random

from termcolor import cprint
import torch
import torch.nn as nn

from utils.arguments import get_args
from kernel_configs import kernel_config

class GaussianFilter(nn.Conv2d):

    def __init__(self, location: int | float = 0, scale: int | float = 1, kernel_size: int = 3, channels: int = 3):
        """Create an (x, y) coordinate grid of shape (kernel_size, kernel_size, 2).

        Args:
            location (int, optional): Distribution location (mu). Defaults to 0.
            scale (int, optional): Distribution scale (sigma). Defaults to 1.
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
        print(f"FILTER NAME: {filter_name}")

        # Set location & scale to 1 if kernel is static (center)
        if filter_name == 'static': location = scale = 1
        print(f"MU (LOCATION): {location} | SIGMA (SCALE): {scale} | VARIANCE: {scale**2}")

        # Create tensors of variables
        seeds = torch.arange(kernel_size)
        x_grid = seeds.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        # Calculate 2D Gaussian Kernel:
        # This will be the PRODUCT of two Gaussian distributions based
        # on variables defind in x_grid and y_grid
        gaussian_kernel = (
            (1. / (scale * math.sqrt(2 * math.pi))) * torch.exp(-0.5 * ((torch.sum(xy_grid, dim=-1) - location) / scale)**2)
        )
        
        # ORIGINAL
        # (
        #     (1. / (2. * math.pi * (scale**2))) * \
        #     torch.exp(
        #         -(torch.sum(xy_grid, dim=-1) - location)**2. / (2 * (scale**2))
        #     )
        # )

        # Ensure sum of distribution is 1
        gaussian_kernel /= torch.sum(gaussian_kernel)

        if filter_name in ['top-right', 'bottom-left', 'bottom-right']:
            if filter_name == 'bottom-right':
                temp = gaussian_kernel[0].clone()
                gaussian_kernel[0] = gaussian_kernel[2]
                gaussian_kernel[2] = temp

            gaussian_kernel = torch.index_select(
                gaussian_kernel, 
                (0 if filter_name == 'bottom-left' else 1), 
                torch.LongTensor([2, 1, 0]))
            
        print(f"{filter_name.upper()}:\n{gaussian_kernel}")

        # Reshape kernel
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)

        self.weight.data = gaussian_kernel
        self.weight.requires_grad = False

if __name__ == '__main__':
    GaussianFilter()