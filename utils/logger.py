"""Logging utilities."""

import datetime, logging, os, sys

from utils  import ARGS

# Initialize logger
LOGGER =        logging.getLogger("hilo")

# Set logging level
LOGGER.setLevel(ARGS.logging_level)

# Ensure that logging_path exists
os.makedirs(
    f"{ARGS.logging_path}",
    exist_ok =  True
)

# Define console handler
stdout_handler =    logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(ARGS.logging_level)
stdout_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(name)s : %(message)s'))
LOGGER.addHandler(stdout_handler)

# Define file handler
file_handler =      logging.FileHandler(
    f"{ARGS.logging_path}/{ARGS.cmd}.log" if ARGS.cmd != "run-job"
    else f"{ARGS.logging_path}/{ARGS.model}/{ARGS.datset}/{ARGS.distribution}{f'/{ARGS.kernel_type}' if ARGS.distribution else ''}/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)
file_handler.setLevel(ARGS.logging_level)
file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(name)s : %(message)s'))
LOGGER.addHandler(file_handler)