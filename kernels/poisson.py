"""Poisson kernel utilities."""

import math, torch
from scipy      import special

from kernels    import AbstractKernel
from utils      import ARGS, LOGGER

class PoissonKernel(AbstractKernel):
    """Poisson distribution kernel."""

    _logger = LOGGER.getChild('poisson-kernel')

    def pdf(self, rate: float = ARGS.rate) -> torch.Tensor:
        """Calculate Poisson distribution kernel.

        Args:
            rate (float, optional): Poisson distribution rate parameter (Lambda). Defaults to ARGS.rate.

        Returns:
            torch.Tensor: Poisson distribution kernel
        """
        self._logger(f"Calculating Poisson distribution (rate: {rate})")

        return ((rate**torch.sum(self.xy_grid, dim=-1)) * (math.e**(-rate))) / (special.factorial(torch.sum(self.xy_grid, dim=-1)))