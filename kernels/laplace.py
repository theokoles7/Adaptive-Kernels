"""Laplace kernel utilities."""

import torch

from kernels    import AbstractKernel
from utils      import ARGS, LOGGER

class LaplaceKernel(AbstractKernel):
    """Laplace distribution kernel."""

    _logger = LOGGER.getChild('laplace-kernel')

    def pdf(self, location: float = ARGS.location, scale: float = ARGS.scale) -> torch.Tensor:
        """Calculate Laplace distribution kernel.

        Args:
            location (float, optional): Laplace distribution location parameter (Mu). Defaults to ARGS.location.
            scale (float, optional): Laplace distribution scale parameter (Sigma). Defaults to ARGS.scale.

        Returns:
            torch.Tensor: Laplace distribution kernel
        """
        self._logger(f"Calculating Laplace distribution (location: {location}, scale {scale})")

        return (
            (1. / (2. * scale)) * (
                torch.exp(
                    -abs(torch.sum(self.xy_grid, dim=-1) - location) / scale
                )
            )
        )