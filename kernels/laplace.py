"""Laplace kernel utilities."""

import torch

from kernels    import AbstractKernel
from utils      import ARGS, LOGGER

class LaplaceKernel(AbstractKernel):
    """Laplace distribution kernel."""

    _logger = LOGGER.getChild('laplace-kernel')

    def __init__(self, location: float, scale: float, channels: int = 3, size: int = ARGS.kernel_size):
        """Initialize Laplace kernel.

        Args:
            location (float): Laplace distribution location parameter (Mu)
            scale (float): Laplace distribution scale parameter (Sigma)
            channels (int, optional): Input channels. Defaults to 3.
            size (int, optional): Kernel size (square). Defaults to ARGS.kernel_size.
        """
        self._location =    location
        self._scale =       scale

        super().__init__(channels, size)

    def pdf(self, xy_grid: torch.Tensor) -> torch.Tensor:
        """Calculate Laplace distribution kernel.

        Args:
            xy_grid (torch.Tensor): XY coordinate grid made from convoluted data

        Returns:
            torch.Tensor: Laplace distribution kernel
        """
        self._logger.info(f"Calculating Laplace distribution (location: {self._location}, scale {self._scale})")

        return (
            (1. / (2. * self._scale)) * (
                torch.exp(
                    -abs(torch.sum(xy_grid, dim=-1) - self._location) / self._scale
                )
            )
        )