"""Cauchy distribution utilities."""

import math, torch

from kernels    import AbstractKernel
from utils      import ARGS, LOGGER

class CauchyKernel(AbstractKernel):
    """Cauchy distribution kernel."""

    _logger = LOGGER.getChild('cauchy-kernel')

    _location:  float
    _scale:     float

    def __init__(self, location: float, scale: float, channels: int = 3, size: int = ARGS.kernel_size):
        """Initialize Cauchy kernel.

        Args:
            location (float): Cauchy distribution location parameter (Chi)
            scale (float): Cauchy distirbution scale parameter (Gamma)
            channels (int, optional): Input channels. Defaults to 3.
            size (int, optional): Kernel size (square). Defaults to ARGS.kernel_size.
        """
        self._location =    location
        self._scale =       scale

        super(CauchyKernel, self).__init__(channels, size)

    def pdf(self, xy_grid: torch.Tensor) -> torch.Tensor:
        """Calculate Cauchy distribution kernel.

        Args:
            xy_grid (torch.Tensor): XY coordinate grid made from convoluted data

        Returns:
            torch.Tensor: Cauchy distribution kernel
        """
        self._logger.info(f"Calculating Cauchy distribution (location: {self._location}, scale {self._scale})")
        
        return (
            1. / (
                (math.pi * self._scale) * (1. + (
                    (torch.sum(xy_grid, dim=-1) - self._location) / (self._scale)
                )**2)
            )
        )