"""Drive application."""

import os, traceback

from utils      import ARGS, BANNER, LOGGER

if __name__ == '__main__':

    try:
        # Print lab banner
        LOGGER.info(BANNER)

        # Match command
        match ARGS.cmd:

            # Initialize results file
            case "init-results":    
                from commands       import init_results
                init_results()

            # Run experiment
            case "run-experiment":  
                from commands       import run_experiment
                run_experiment()

            # Run job
            case "run-job":
                from commands.job   import Job
                Job().run()

    except KeyboardInterrupt:
        LOGGER.critical("Keyboard interrupt detected. Aborting operations.")

    except Exception as e:
        LOGGER.error(f"An error occured: {e}")
        traceback.print_exc()

    finally:
        LOGGER.info("Exiting...")