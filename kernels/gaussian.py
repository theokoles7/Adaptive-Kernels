"""Gaussian distribution utilities."""

import math, torch

from kernels    import AbstractKernel
from utils      import ARGS, LOGGER

class GaussianKernel(AbstractKernel):
    """Gaussian distirbution kernel."""

    _logger = LOGGER.getChild('gaussian-kernel')

    def pdf(self, location: float = ARGS.location, scale: float = ARGS.scale) -> torch.Tensor:
        """Calculate Gaussian distribution kernel.

        Args:
            location (float, optional): Gaussian distribution location parameter (Mu). Defaults to ARGS.location.
            scale (float, optional): Gaussian distribution scale parameter (Sigma). Defaults to ARGS.scale.

        Returns:
            torch.Tensor: Gaussian distribution kernel
        """
        self._logger(f"Calculating Gaussian distribution (location: {location}, scale {scale})")

        return (
            (1. / (2. * math.pi * (scale)**2)) * torch.exp(
                -torch.sum(
                    (self.xy_grid - location)**2., dim=-1
                ) /\
                (2 * (scale)**2)
            )
        )