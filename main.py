"""Drive application."""

from commands   import run_experiment, run_job
from utils      import ARGS, BANNER, LOGGER

if __name__ == '__main__':
    """Execute application command."""

    try:# Print lab banner
        LOGGER.info(BANNER)

        # Match command
        match ARGS.cmd:

            # Run experiment
            case "run-experiment":  run_experiment(**vars(ARGS))

            # Run job
            case "run-job":         run_job(**vars(ARGS))

    # Exit gracefully on keyboard interruptions
    except KeyboardInterrupt:   LOGGER.critical("Keyboard interrupt detected. Aborting operations.")

    # Log errors that are not accounted for
    except Exception as e:      LOGGER.error(f"An error occured: {e}", exc_info = True)

    # Exit gracefully
    finally:                    LOGGER.info("Exiting...")