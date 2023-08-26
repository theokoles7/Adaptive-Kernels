"""Kernel accessor."""

from utils.globals import ARGS

from kernels.cauchy import CauchyKernel
from kernels.gaussian import GaussianKernel
from kernels.gumbel import GumbelKernel
from kernels.laplace import LaplaceKernel
from kernels.poisson import PoissonKernel

def get_kernel( channels: int, location: float = None, scale: float = None, rate: float = None):
    """Provide kernel based on selected distribution.

    Args:
        channels (int): Model layer channels (input & output)
        location (float, optional): Location parameter for bivariate distributions. Defaults to None.
        scale    (float, optional): Scale parameter for bivariate distributions. Defaults to None.
        rate     (float, optional): Rate parameter for univariate distribution (Poisson). Defaults ot None.
    """
    match ARGS.distribution:
        case 'cauchy':      return CauchyKernel(  location=location, scale=scale, channels=channels)
        case 'gaussian':    return GaussianKernel(location=location, scale=scale, channels=channels)
        case 'gumbel':      return GumbelKernel(  location=location, scale=scale, channels=channels)
        case 'laplace':     return LaplaceKernel( location=location, scale=scale, channels=channels)
        case 'poisson':     return PoissonKernel( rate=rate,                      channels=channels)

        case _: raise NotImplementedError(f"{ARGS.distributino} not supported")