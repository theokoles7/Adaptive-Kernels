"""Gumbel kernel utilities."""

from typing     import override

from torch      import exp, sum, Tensor

from kernels    import Kernel

class GumbelKernel(Kernel):
    """# Gumbel distribution kernel."""

    @override
    def pdf(self,
        xy_grid:    Tensor
    ) -> Tensor:
        """# Calculate Gumbel distribution kernel.

        ## Args:
            * xy_grid   (Tensor):   XY coordinate grid made from convoluted data.

        ## Returns:
            * Tensor:   Gumbel distribution kernel.
        """
        # Log for debugging
        self.__logger__.info(f"Calculating Gumbel distribution (location: {self._location_}, scale {self._scale_})")

        # Calculate Gumbel kernel
        return (
            (1 / self._scale_) * exp(
                -(
                    ((sum(xy_grid - self._location_, dim=-1)) / self._scale_) +\
                        (exp(((sum(xy_grid - self._location_, dim=-1)) / self._scale_)))
                )
            )
        )