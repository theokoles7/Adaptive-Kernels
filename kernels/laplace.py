"""Laplace kernel utilities."""

from typing     import override

from torch      import exp, sum, Tensor

from kernels    import Kernel

class LaplaceKernel(Kernel):
    """Laplace distribution kernel."""

    @override
    def pdf(self,
        xy_grid:    Tensor
    ) -> Tensor:
        """# Calculate Laplace distribution kernel.

        ## Args:
            * xy_grid   (Tensor):   XY coordinate grid made from convoluted data.

        ## Returns:
            * Tensor:   Laplace distribution kernel.
        """
        # Log for debugging
        self.__logger__.info(f"Calculating Laplace distribution (location: {self._location_}, scale {self._scale_})")

        # Calculate Laplace kernel
        return (
            (1. / (2. * self._scale_)) * (
                exp(
                    -abs(sum(xy_grid, dim=-1) - self._location_) / self._scale_
                )
            )
        )