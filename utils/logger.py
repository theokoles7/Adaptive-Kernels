"""Logger utilities."""

import datetime, logging, os, sys

class Logger():
    """Logger class."""

    def __init__(self, path: str, to_file: bool = True):
        """Initialize Logger.

        Args:
            path (str): Logger output path
        """
        # Initialize logger
        self._logger = logging.getLogger()

        # Set general logger level
        self._logger.setLevel(logging.DEBUG)

        # Define console handler
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.DEBUG)
        stdout_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
        self._logger.addHandler(stdout_handler)

        if to_file:
            # Verify that path exists
            if not os.path.exists(path): os.makedirs(path)

            # Define file handler
            file_handler = logging.FileHandler(f"{path}/{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
            self._logger.addHandler(file_handler)

    def get_logger(self) -> logging.Logger:
        """Provide logger.

        Returns:
            logging.Logger: Logger object
        """
        return self._logger