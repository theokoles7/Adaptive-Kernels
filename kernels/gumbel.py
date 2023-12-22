"""Gumbel kernel utilities."""

import torch

from kernels    import AbstractKernel
from utils      import ARGS, LOGGER

class GumbelKernel(AbstractKernel):
    """Gumbel distribution kernel."""

    _logger = LOGGER.getChild('gumbel-kernel')

    def pdf(self, location: float = ARGS.location, scale: float = ARGS.scale) -> torch.Tensor:
        """Calculate Gumbel distribution kernel.

        Args:
            location (float, optional): Gumbel distribution location parameter (Mu). Defaults to ARGS.location.
            scale (float, optional): Gumbel distribution scale parameter (Beta). Defaults to ARGS.scale.

        Returns:
            torch.Tensor: Gumbel distribution kernel
        """
        self._logger(f"Calculating Gumbel distribution (location: {location}, scale {scale})")

        return (
            (1 / scale) * torch.exp(
                -(
                    ((torch.sum(self.xy_grid - location, dim=-1)) / scale) +\
                        (torch.exp(((torch.sum(self.xy_grid - location, dim=-1)) / scale)))
                )
            )
        )