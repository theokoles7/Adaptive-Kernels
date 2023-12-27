"""Resnet model block."""

import torch, torch.nn as nn, torch.nn.functional as F

from kernels    import get_kernel
from utils      import ARGS, LOGGER

class ResnetBlock(nn.Module):
    """Block component for Resnet Model."""

    # Initialize logger
    _logger =       LOGGER.getChild('resnet-block')

    def __init__(self, channels_in: int, channels_out: int, stride: int):
        """Initialize Resnet block.

        Args:
            channels_in (int): Input channels
            channels_out (int): Output channels
            stride (int): Convolution stride
        """
        super(ResnetBlock, self).__init__()

        # Initialize distribution parameters
        self.location =     [ARGS.location if ARGS.distribtion != "poisson" else ARGS.rate]*2
        self.scale =        [ARGS.scale]*2

        self.channels_out = channels_out

        # Batch normalization layers
        self.bn1 =          nn.BatchNorm2d(channels_out)
        self.bn2 =          nn.BatchNorm2d(channels_out)

        # Convolving layers
        self.conv1 =        nn.Conv2d(channels_in,  channels_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 =        nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1,      padding=1, bias=False)

        # Shortcut layer
        self.shortcut =     nn.Sequential()

        # if stride != 1 or channels_in != (channels_out):
        #     self.shortcut_kernel = True
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(channels_in, (channels_out), kernel_size=1, stride=stride, bias=False),
        #         nn.BatchNorm2d(channels_out)
        #     )

        def forward(self, X: torch.Tensor) -> torch.Tensor:
            """Feed input through network and provide output.

            Args:
                X (torch.Tensor): Input tensor

            Returns:
                torch.Tensor: Output tensor
            """
            # INPUT LAYER =========================================================================
            self._logger.debug(f"Input shape: {X.shape}")

            # LAYER 1 =============================================================================
            x1 = self.conv1(X)

            self._logger.debug(f"Layer 1 shape: {x1.shape}")

            with torch.no_grad(): y, self.location[0], self.scale[0] = x1.float(), torch.mean(y).item(), torch.std(y).item()

            # LAYER 2 =============================================================================
            x2 = self.conv2(F.relu(self.bn1(self.kernel1(x1) if ARGS.distribution else x1)))

            self._logger.debug(f"Layer 2 shape: {x2.shape}")

            with torch.no_grad(): y, self.location[1], self.scale[1] = x2.float(), torch.mean(y).item(), torch.std(y).item()

            # OUTPUT LAYER ========================================================================
            output = self.bn2(self.kernel2(x2) if ARGS.distribution else x2)

            self._logger.debug(f"Output layer shape: {output.shape}")

            # SHORTCUT LAYER ======================================================================
            output += self.shortcut(output)

            # Return output
            return F.relu(output)
        
        def set_kernels(self) -> None:
            """Create/update kernels."""
            self.kernel1 = get_kernel(ARGS.distribution, ARGS.kernel_size, self.channels_out, location=self.location[0], scale=self.scale[0], rate=self.location[0])
            self.kernel2 = get_kernel(ARGS.distribution, ARGS.kernel_size, self.channels_out, location=self.location[1], scale=self.scale[1], rate=self.location[1])