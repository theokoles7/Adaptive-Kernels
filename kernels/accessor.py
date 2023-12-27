"""Kernel accessor."""

from kernels    import CauchyKernel, GaussianKernel, GumbelKernel, LaplaceKernel, PoissonKernel

from utils      import ARGS, LOGGER

logger = LOGGER.getChild('kernel-accessor')

def get_kernel(distribution: str, size: int, channels: int, location: float, scale: float, rate: float) -> CauchyKernel | GaussianKernel | GumbelKernel | LaplaceKernel | PoissonKernel:
    """Provide appropriate distribution kernel.

    Args:
        distribution (str, optional): Distribution selection
        size (int, optional): Kernel size (square)
        channels (int, optional): Input channels
        location (float, optional): Distribution location parameter
        scale (float, optional): Distribution scale parameter
        rate (float, optional): Distirbution rate parameter

    Returns:
        CauchyKernel | GaussianKernel | GumbelKernel | LaplaceKernel | PoissonKernel: Selected distribution kernel
    """
    logger.info(f"Fetching kernel: {ARGS.distribution}")

    match distribution:

        case 'cauchy':      return CauchyKernel(    location, scale, channels, size)
        case 'gaussian':    return GaussianKernel(  location, scale, channels, size)
        case 'gumbel':      return GumbelKernel(    location, scale, channels, size)
        case 'laplace':     return LaplaceKernel(   location, scale, channels, size)
        case 'poisson':     return PoissonKernel(   rate,            channels, size)

        case _:
            raise NotImplementedError(
                f"{distribution} kernel not supported"
            )