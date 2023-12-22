"""Abstract kernel class and utilities."""

import math, random, torch, torch.nn as nn

from utils  import ARGS

class AbstractKernel(nn.Conv2d):
    """Abstract kernel class."""

    __kernel_config = {
        1:  ['top-left',    'bottom-right'],
        2:  ['bottom-left', 'top-right'],
        3:  ['top-left',    'bottom-left'],
        4:  ['top-right',   'bottom-right'],
        5:  ['top-left',    'top-right'],
        6:  ['bottom-left', 'bottom-right'],
        7:  ['top-left',    'bottom-right', 'static'],
        8:  ['bottom-left', 'top-right',    'static'],
        9:  ['top-left',    'bottom-left',  'static'],
        10: ['top-right',   'bottom-right', 'static'],
        11: ['top-left',    'top-right',    'static'],
        12: ['bottom-left', 'bottom-right', 'static'],
        13: ['top-left',    'bottom-right', 'bottom-left', 'top-right'],
        14: ['top-left',    'bottom-right', 'bottom-left', 'top-right', 'static']
    }

    def __init__(self, channels: int = 3, size: int = ARGS.kernel_size):
        """Create an (x, y) coordinate grid using convoluted data.

        Args:
            channels (int, optional): Input channels. Defaults to 3.
            size (int, optional): Kernel size (square). Defaults to ARGS.kernel_size.
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
        filter_list =   self.__kernel_config[ARGS.kernel_type]
        self._logger.debug(f"KERNEL TYPE {ARGS.kernel_type}: {filter_list}")

        # Randomly select primary filter from list
        filter_name =   filter_list[random.randint(0, len(filter_list) - 1)]
        self._logger.debug(f"FILTER NAME: {filter_name}")

        # Create tensors
        seeds =         torch.arange(size)
        x_grid =        seeds.repeat(size).view(size, size)
        y_grid =        x_grid.t()
        xy_grid =       torch.stack([x_grid, y_grid], dim=-1).float()

        # Calculate kernel
        kernel = self.pdf()

        # Ensure kernel does not contain NAN or INF
        assert not torch.isinf(kernel).any(), f"NAN or INF detected in kernel:\n{kernel}"

        # Ensure sum of distribution is 1
        kernel /= torch.sum(kernel)

        if filter_name in ['top-right', 'bottom-left', 'bottom-right']:

            if filter_name == 'bottom-right':
                # Swap first & last kernels
                temp =          kernel[0].clone()
                kernel[0] =     kernel[-1]
                kernel[-1] =    temp

            kernel = torch.index_select(
                kernel,
                (0 if filter_name == 'bottom-left' else 1),
                torch.LongTensor([2, 1, 0])
            )

        self._logger.info(f"{filter_name.upper()}:\n{kernel}")

        # Reshape kernel
        kernel =    kernel.view(1, 1, size, size).repeat(channels, 1, 1, 1)

        self.weight.data =          kernel
        self.weight.requires_grad = False