"""Drive application."""

import os, traceback

from commands   import init_results, run_experiment, Job
from utils      import ARGS, BANNER, LOGGER

if __name__ == '__main__':

    try:
        # Print lab banner
        LOGGER.info(BANNER)

        # Match command
        match ARGS.cmd:

            # Initialize results file
            case "init-results":    init_results()

            # Run experiment
            case "run-experiment":  run_experiment()

            # Run job
            case "run-job":         Job().run()

    except KeyboardInterrupt:
        LOGGER.critical("Keyboard interrupt detected. Aborting operations.")

    except Exception as e:
        LOGGER.error(f"An error occured: {e}")
        traceback.print_exc()

    finally:
        LOGGER.info("Exiting...")