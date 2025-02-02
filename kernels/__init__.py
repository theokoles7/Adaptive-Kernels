__all__ = ["__base__", "cauchy", "gaussian", "gumbel", "laplace"]

from kernels.__base__           import Kernel

from kernels.cauchy             import CauchyKernel
from kernels.gaussian           import GaussianKernel
from kernels.gumbel             import GumbelKernel
from kernels.laplace            import LaplaceKernel

def load_kernel(
    kernel:     str,
    **kwargs
) -> Kernel:
    """# Load specified kernel.

    ## Args:
        * kernel    (str):  Kernel selection.

    ## Returns:
        * Kernel:   Selected kernel, initialized and ready for convolving.
    """
    # Match kernel selection
    match kernel:
        
        # Cauchy
        case "cauchy":      return CauchyKernel(**kwargs)
        
        # Gaussian
        case "gaussian":    return GaussianKernel(**kwargs)
        
        # Gumbel
        case "gumbel":      return GumbelKernel(**kwargs)
        
        # Laplace
        case "laplace":     return LaplaceKernel(**kwargs)
        
        # Invalid selection
        case _:             raise NotImplementedError(f"Invalid kernel selection: {kernel}")