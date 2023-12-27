"""Poisson kernel utilities."""

import math, torch
from scipy      import special

from kernels    import AbstractKernel
from utils      import ARGS, LOGGER

class PoissonKernel(AbstractKernel):
    """Poisson distribution kernel."""

    _logger = LOGGER.getChild('poisson-kernel')

    def __init__(self, rate: float, channels: int = 3, size: int = ARGS.kernel_size):
        """Initialize Poisson kernel.

        Args:
            rate (float): Poisson distribution rate parameter (Lambda)
            channels (int, optional): Input channels. Defaults to 3.
            size (int, optional): Kernel size (square). Defaults to ARGS.kernel_size.
        """
        self._rate =    rate

        super().__init__(channels, size)

    def pdf(self, xy_grid: torch.Tensor) -> torch.Tensor:
        """Calculate Poisson distribution kernel.

        Args:
            xy_grid (torch.Tensor): XY coordinate grid made from convoluted data

        Returns:
            torch.Tensor: Poisson distribution kernel
        """
        self._logger.info(f"Calculating Poisson distribution (rate: {self._rate})")

        return ((self._rate**torch.sum(xy_grid, dim=-1)) * (math.e**(-self._rate))) / (special.factorial(torch.sum(xy_grid, dim=-1)))