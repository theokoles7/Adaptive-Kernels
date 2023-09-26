"""Distribution kernel utilities."""

from termcolor import colored

from utils import ARGS, LOGGER

from kernels.cauchy     import CauchyKernel
from kernels.gaussian   import GaussianKernel
from kernels.gumbel     import GumbelKernel
from kernels.laplace    import LaplaceKernel
from kernels.poisson    import PoissonKernel

# Initialize kernel accessor logger
logger = LOGGER.getChild('kernel-accessor')

def get_kernel(
        size: int,
        channels: int,
        location: float = None,
        scale: float = None,
        rate: float = None
    ) -> CauchyKernel | GaussianKernel | GumbelKernel | LaplaceKernel | PoissonKernel:
    """Provide appropriate distribution kernel.

    Args:
        size (int): Kernel size (square).
        channels (int, optional): Input channels. Defaults to 3.
        location (float, optional): Location parameter (mu) of Gaussian distribution. Defaults to None.
        scale (float, optional): Scale parameter (sigma) of Gaussian distribution. Defaults to None.

    Returns:
        CauchyKernel | GaussianKernel | GumbelKernel | LaplaceKernel | PoissonKernel: Selected distribution kernel
    """
    logger.info(f"Fetching kernel: {ARGS.distribution}")

    match ARGS.distribution:
        case 'cauchy':      return CauchyKernel(    location, scale, channels, size)
        case 'gaussian':    return GaussianKernel(  location, scale, channels, size)
        case 'gumbel':      return GumbelKernel(    location, scale, channels, size)
        case 'laplace':     return LaplaceKernel(   location, scale, channels, size)
        case 'poisson':     return PoissonKernel(   rate,            channels, size)

        case _:
            raise NotImplementedError(
                f"{ARGS.distribution} not supported."
            )
        
if __name__ == '__main__':
    "Test kernel distribution accessor."

    # Access Cauchy kernel
    ARGS.distribution = 'cauchy'
    print(f"\n{colored('Access Cauchy kernel', 'green')}: {get_kernel(3, 3, location=0, scale=1)}")

    # Access Gaussian kernel
    ARGS.distribution = 'gaussian'
    print(f"\n{colored('Access Gaussian kernel', 'green')}: {get_kernel(3, 3, location=0, scale=1)}")

    # Access Gumbel kernel
    ARGS.distribution = 'gumbel'
    print(f"\n{colored('Access Gumbel kernel', 'green')}: {get_kernel(3, 3, location=0, scale=1)}")

    # Access Laplace kernel
    ARGS.distribution = 'laplace'
    print(f"\n{colored('Access Laplace kernel', 'green')}: {get_kernel(3, 3, location=0, scale=1)}")

    # Access Poisson kernel
    ARGS.distribution = 'poisson'
    print(f"\n{colored('Access Poisson kernel', 'green')}: {get_kernel(3, 3, rate=1)}")