"""Wrapper methods and commonly used functions."""

import arguments, math, random
import torch
import torch.nn as nn

from termcolor import cprint

from kernel_configs import kernel_config

def get_gaussian_filter(epoch_number: int, mean: int = 1, kernel_size: int = 3, sigma: int = 2, channels: int = 3) -> nn.Conv2d:
    """Create an (x, y) coordinate grid of shape (kernel_size, kernel_size, 2).

    Args:
        epoch_number (int): Number of epochs.
        mean (int, optional): Mu; Distribution mean. Defaults to 1.
        sigma (int, optional): Sigma: Distribution standard deviation. Defaults to 2.
        kernel_size (int, optional): Kernel dimension. Defaults to 3.
        channels (int, optional): Input channels. Defaults to 3.

    Returns:
        nn.Conv2d: Gaussian distribution-based kernel.
    """
    # Verify integrity of mean and sigma data types
    if math.isnan(mean): raise ValueError(f"Mean expected to be of type int; Got {type(mean)}")
    if math.isnan(sigma): raise ValueError(f"Sigma expected to be of type int; Got {type(sigma)}")

    # Parse arguments
    args = arguments.get_args()

    # Fetch corresponding filter list for kernel type provided
    filter_list = kernel_config[args.kernel_type]
    print(f"Kernel Type {args.kernel_type}: {filter_list}")

    filter_name = filter_list[random.randint(0,len(filter_list)-1)]
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (1 if filter_name == 'static' else mean)
    sigma = (1 if filter_name == 'static' else sigma)

    variance = sigma**2.

    cprint(f"Sigma: {sigma} --Variance: {variance} --mean: {mean}", "red", "on_white")

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    if filter_name == 'static':
        cprint("static", "red", "on_white")
        print(gaussian_kernel)

    elif filter_name == 'topLeft':
        cprint("Top Left", "red", "on_white")
        print(gaussian_kernel)

    elif filter_name == 'bottomRight':
        temp = gaussian_kernel[0].clone()
        gaussian_kernel[0] = gaussian_kernel[2]
        gaussian_kernel[2] = temp
        idx = torch.LongTensor([2, 1, 0])
        gaussian_kernel = torch.index_select(gaussian_kernel, 1, idx)
        cprint("Bottom Right", "red", "on_white")
        print(gaussian_kernel)

    elif filter_name == 'topRight':
        idx = torch.LongTensor([2, 1, 0])
        gaussian_kernel = torch.index_select(gaussian_kernel, 1, idx)
        cprint("Top Right", "red", "on_white")
        print(gaussian_kernel)

    elif filter_name == 'bottomLeft':
        idx = torch.LongTensor([2, 1, 0])
        gaussian_kernel = torch.index_select(gaussian_kernel, 0, idx)
        cprint("Bottom Left", "red", "on_white")
        print(gaussian_kernel)










    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
    if kernel_size == 3:
        padding = 1
    elif kernel_size == 5:
        padding = 2
    else:
        padding = 0

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels,
                                bias=False, padding=padding)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    
    return gaussian_filter
