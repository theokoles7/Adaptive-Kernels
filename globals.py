"""Global variables."""

from utils.arguments import Arguments
from utils.logger import Logger

ARGS = Arguments().get_args()
LOGGER = Logger(ARGS.logger_path, to_file=False).get_logger()

if ARGS.debug: LOGGER.debug(Arguments())