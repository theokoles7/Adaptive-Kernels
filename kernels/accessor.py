"""Kernel accessor."""

from utils.globals import ARGS

from kernels.cauchy import CauchyKernel
from kernels.gaussian import GaussianKernel
from kernels.gumbel import GumbelKernel
from kernels.laplace import LaplaceKernel
from kernels.poisson import PoissonKernel

def get_kernel(channels: int):
    """Provide kernel based on selected distribution.

    Args:
        channels (int): Model layer channels (input & output)
    """
    match ARGS.distribution:
        case 'cauchy':      return CauchyKernel(channels)
        case 'gaussian':    return GaussianKernel(channels)
        case 'gumbel':      return GumbelKernel(channels)
        case 'laplace':     return LaplaceKernel(channels)
        case 'poisson':     return PoissonKernel(channels)

        case _: raise NotImplementedError(f"{ARGS.distributino} not supported")