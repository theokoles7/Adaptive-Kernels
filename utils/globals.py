"""Global variables."""

from utils.arguments import Arguments
from utils.logger import Logger

# UTILITIES
ARGS = Arguments().get_args()                   # Command line arguments
LOGGER = Logger(ARGS.logger_path).get_logger()  # Logger

if ARGS.debug: LOGGER.debug(Arguments())

# DATASET
NUM_CLASSES = 0
CHANNELS_IN = 0