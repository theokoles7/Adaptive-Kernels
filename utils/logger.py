"""Logging utiliites."""

import datetime, logging, os, sys

class Logger():
    """Logger class."""

    def __init__(self, name: str, path: str):
        """Initialize Logger object.

        self.ARGS:
            name (str): Logger name
            path (str): Logger output path
        """
        # Initialize logger
        self._logger = logging.getLogger(name)

        # Set logging level
        self._logger.setLevel(logging.DEBUG)

        # Ensure that output path exists
        os.makedirs(path, exist_ok=True)

        # Define console handler
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.DEBUG)
        stdout_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(name)s : %(message)s'))
        self._logger.addHandler(stdout_handler)

        # Define file handler
        file_handler = logging.FileHandler(f"{path}/{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(name)s : %(message)s'))
        self._logger.addHandler(file_handler)

    def get_logger(self) -> logging.Logger:
        """Provide logger.

        Returns:
            logging.Logger: Logger object
        """
        return self._logger
    
if __name__ == '__main__':
    """Test logger."""
    # Initialize logger
    logger = Logger('test', 'test').get_logger()

    # Test logging
    logger.info("This is an info message.")
    logger.debug("This is a debug message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")