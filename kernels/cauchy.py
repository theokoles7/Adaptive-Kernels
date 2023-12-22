"""Cauchy distribution utilities."""

import math, torch

from kernels    import AbstractKernel
from utils      import ARGS, LOGGER

class CauchyKernel(AbstractKernel):
    """Cauchy distribution kernel."""

    _logger = LOGGER.getChild('cauchy-kernel')

    def pdf(self, location: float = ARGS.location, scale: float = ARGS.scale) -> torch.Tensor:
        """Calculate Cauchy distribution kernel.

        Args:
            location (float, optional): Cauchy distribution location parameter (Chi). Defaults to ARGS.location.
            scale (float, optional): Cauchy distribution scale parameter (Gamma). Defaults to ARGS.scale.

        Returns:
            torch.Tensor: Cauchy distribution kernel
        """
        self._logger(f"Calculating Cauchy distribution (location: {location}, scale {scale})")
        
        return (
            1. / (
                (math.pi * scale) * (1. + (
                    (torch.sum(self.xy_grid, dim=-1) - location) / (scale)
                )**2)
            )
        )