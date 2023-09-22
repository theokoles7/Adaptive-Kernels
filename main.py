"""Drive Hi-Lo application."""

import os

from utils.logger import Logger
from utils.arguments import Arguments
from utils.banner import *

if __name__ == '__main__':
    # Intialize utilities
    LOGGER =    Logger('test', 'test').get_logger()
    ARGS =      Arguments().get_args()

    try:
        LOGGER.info(lab_banner + arg_banner)

    except Exception as e:
        print(e)
        LOGGER.error(e)