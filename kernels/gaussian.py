"""Gaussian distribution utilities."""

from math       import pi
from typing     import override

from torch      import exp, sum, Tensor

from kernels    import Kernel

class GaussianKernel(Kernel):
    """# Gaussian distirbution kernel."""

    @override
    def pdf(self,
        xy_grid:    Tensor
    ) -> Tensor:
        """# Calculate Gaussian distribution kernel.

        ## Args:
            * xy_grid   (Tensor):   XY coordinate grid made from convoluted data.

        ## Returns:
            * torch.Tensor: Gaussian distribution kernel.
        """
        # Log for debugging
        self.__logger__.debug(f"Calculating Gaussian distribution (location: {self._location_}, scale {self._scale_})")

        # Calculate Gaussian kernel
        return (
            (1. / (2. * pi * (self._scale_)**2)) * exp(
                -sum(
                    (xy_grid - self._location_)**2., dim=-1
                ) /\
                (2 * (self._scale_)**2)
            )
        )