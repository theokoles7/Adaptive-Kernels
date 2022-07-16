import math

import torch
import torch.nn as nn
from termcolor import colored,cprint



def get_gaussian_filter(epoch_number,mean =1, kernel_size=3, sigma=2, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    filter_list = ['dataStd', 'forceG', 'invDataStd']
    filter_name = ""
    if epoch_number % 2 == 0 and epoch_number % 3 != 0:
        filter_name = filter_list[1]
    elif epoch_number % 3 == 0:
        filter_name = filter_list[2]
    else:
        filter_name = filter_list[0]

    cprint("Epoch Number:" + str(epoch_number) + " filter name: " + filter_name, "red", "on_white")



    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    if filter_name == filter_list[1]:
        mean = 1
        sigma = 1
    else:
        mean = mean
        sigma = sigma

    variance = sigma**2.




    cprint("Sigma: " + str(sigma) + " --Variance: " + str(variance) + " --mean: " + str(mean), "red", "on_white")

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
    print(gaussian_kernel)

    if filter_name == filter_list[2]:
        temp = gaussian_kernel[0].clone()
        gaussian_kernel[0] = gaussian_kernel[2]
        gaussian_kernel[2] = temp
        idx = torch.LongTensor([2, 1, 0])
        gaussian_kernel = torch.index_select(gaussian_kernel, 1, idx)
        cprint("inverse data Std", "red", "on_white")
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
