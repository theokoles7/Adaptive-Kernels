"""Kernel accessor."""

from kernels    import CauchyKernel, GaussianKernel, GumbelKernel, LaplaceKernel, PoissonKernel

from utils      import ARGS, LOGGER

logger = LOGGER.getChild('kernel-accessor')

def get_kernel(distribution: str = ARGS.distribution, size: int = ARGS.kernel_size, channels: int = ARGS.channels) -> CauchyKernel | GaussianKernel | GumbelKernel | LaplaceKernel | PoissonKernel:
    """Provide appropriate distribution kernel.

    Args:
        distribution (str, optional): Distribution selection. Defaults to ARGS.distribution.
        size (int, optional): Kernel size (square). Defaults to ARGS.kernel_size.
        channels (int, optional): Input channels. Defaults to ARGS.channels.

    Returns:
        CauchyKernel | GaussianKernel | GumbelKernel | LaplaceKernel | PoissonKernel: Selected distribution kernel
    """
    logger.info(f"Fetching kernel: {ARGS.distribution}")

    match distribution:

        case 'cauchy':      return CauchyKernel(    ARGS.location, ARGS.scale, channels, size)
        case 'gaussian':    return GaussianKernel(  ARGS.location, ARGS.scale, channels, size)
        case 'gumbel':      return GumbelKernel(    ARGS.location, ARGS.scale, channels, size)
        case 'laplace':     return LaplaceKernel(   ARGS.location, ARGS.scale, channels, size)
        case 'poisson':     return PoissonKernel(   ARGS.rate,                 channels, size)

        case _:
            raise NotImplementedError(
                f"{distribution} kernel not supported"
            )