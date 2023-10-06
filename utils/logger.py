"""Logging utiliites."""

import datetime, logging, os, sys

from utils.arguments import ARGS

# Initialize logger
LOGGER = logging.getLogger('hi-lo')

# Set logging level
LOGGER.setLevel(logging.DEBUG)

# Ensure that output ARGS.logger_path exists
os.makedirs(ARGS.logger_path, exist_ok=True)

# Define console handler
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(name)s : %(message)s'))
LOGGER.addHandler(stdout_handler)

# Define file handler
file_handler = logging.FileHandler(f"{ARGS.logger_path}/{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(name)s : %(message)s'))
LOGGER.addHandler(file_handler)