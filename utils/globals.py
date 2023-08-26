"""Global variables."""

from utils.arguments import Arguments
from utils.logger import Logger

# UTILITIES
ARGS = Arguments().get_args()                   # Command line arguments
LOGGER = Logger(f"{ARGS.logger_path}/{ARGS.distribution}{f'/{ARGS.kernel_type}' if ARGS.distribution else ''}/{ARGS.model}/{ARGS.dataset}").get_logger()  # Logger

if ARGS.debug: LOGGER.debug(Arguments())