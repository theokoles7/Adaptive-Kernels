__all__ = ["cauchy", "gaussian", "gumbel", "laplace"]

from utils.arguments.kernels.cauchy     import add_cauchy_parser
from utils.arguments.kernels.gaussian   import add_gaussian_parser
from utils.arguments.kernels.gumbel     import add_gumbel_parser
from utils.arguments.kernels.laplace    import add_laplace_parser