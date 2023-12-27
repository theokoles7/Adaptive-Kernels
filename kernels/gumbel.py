"""Gumbel kernel utilities."""

import torch

from kernels    import AbstractKernel
from utils      import ARGS, LOGGER

class GumbelKernel(AbstractKernel):
    """Gumbel distribution kernel."""

    _logger = LOGGER.getChild('gumbel-kernel')

    def __init__(self, location: float, scale: float, channels: int = 3, size: int = ARGS.kernel_size):
        """Initialize Gumbel kernel.

        Args:
            location (float): Gumbel distribution location parameter (Mu)
            scale (float): Gumbel distribution scale parameter (Beta)
            channels (int, optional): Input channels. Defaults to 3.
            size (int, optional): Kernel size (square). Defaults to ARGS.kernel_size.
        """
        self._location =    location
        self._scale =       scale

        super().__init__(channels, size)

    def pdf(self, xy_grid: torch.Tensor) -> torch.Tensor:
        """Calculate Gumbel distribution kernel.

        Args:
            xy_grid (torch.Tensor): XY coordinate grid made from convoluted data

        Returns:
            torch.Tensor: Gumbel distribution kernel
        """
        self._logger.info(f"Calculating Gumbel distribution (location: {self._location}, scale {self._scale})")

        return (
            (1 / self._scale) * torch.exp(
                -(
                    ((torch.sum(xy_grid - self._location, dim=-1)) / self._scale) +\
                        (torch.exp(((torch.sum(xy_grid - self._location, dim=-1)) / self._scale)))
                )
            )
        )