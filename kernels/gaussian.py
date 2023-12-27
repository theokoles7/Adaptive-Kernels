"""Gaussian distribution utilities."""

import math, torch

from kernels    import AbstractKernel
from utils      import ARGS, LOGGER

class GaussianKernel(AbstractKernel):
    """Gaussian distirbution kernel."""

    _logger = LOGGER.getChild('gaussian-kernel')

    def __init__(self, location: float, scale: float, channels: int = 3, size: int = ARGS.kernel_size):
        """Initialize Gaussian kernel.

        Args:
            location (float): Gaussian distribution location parameter (Mu)
            scale (float): Gaussian distribution scale parameter (Sigma)
            channels (int, optional): Input channels. Defaults to 3.
            size (int, optional): Kernel size (square). Defaults to ARGS.kernel_size.
        """
        self._location =    location
        self._scale =       scale

        super().__init__(channels, size)

    def pdf(self, xy_grid: torch.Tensor) -> torch.Tensor:
        """Calculate Gaussian distribution kernel.

        Args:
            xy_grid (torch.Tensor): XY coordinate grid made from convoluted data

        Returns:
            torch.Tensor: Gaussian distribution kernel
        """
        self._logger.info(f"Calculating Gaussian distribution (location: {self._location}, scale {self._scale})")

        return (
            (1. / (2. * math.pi * (self._scale)**2)) * torch.exp(
                -torch.sum(
                    (xy_grid - self._location)**2., dim=-1
                ) /\
                (2 * (self._scale)**2)
            )
        )