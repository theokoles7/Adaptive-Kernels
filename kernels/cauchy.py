"""Cauchy distribution utilities."""

from math       import pi
from typing     import override

from torch      import sum, Tensor

from kernels    import Kernel

class CauchyKernel(Kernel):
    """# Cauchy distribution kernel."""

    @override
    def pdf(self,
        xy_grid:    Tensor
    ) -> Tensor:
        """# Calculate Cauchy distribution kernel.

        ## Args:
            * xy_grid   (Tensor):   XY coordinate grid made from convoluted data.

        ## Returns:
            * Tensor:   Cauchy distribution kernel.
        """
        # Log for debugging
        self.__logger__.debug(f"Calculating Cauchy distribution (location: {self._location_}, scale {self._scale_})")
        
        # Calculate Cauchy kernel
        return (
            1. / (
                (pi * self._scale_) * (1. + (
                    (sum(xy_grid, dim=-1) - self._location_) / (self._scale_)
                )**2)
            )
        )