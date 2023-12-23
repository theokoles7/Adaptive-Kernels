"""Run predetermined experiments."""

import os

from utils  import ARGS, LOGGER

def run_experiment() -> None:
    """Execute bash script outlining predetermined experiment."""

    # Determine if experiment is being run by model, dataset, or kernel
    if ARGS.by_model:
        LOGGER.info(f"Executing experiment by model: {ARGS.by_model}")

        os.system(f"./experiments/by_model/{ARGS.by_model}.sh")

    elif ARGS.by_kernel:
        LOGGER.info(f"Executing experiment by kernel: {ARGS.by_kernel}")

        os.system(f"./experiments/by_kernel/{ARGS.by_kernel}.sh")

    elif ARGS.by_dataset:
        LOGGER.info(f"Executing experiment by dataset: {ARGS.by_dataset}")

        os.system(f"./experiments/by_dataset/{ARGS.dataset}.sh")
            
    elif ARGS.control:
        LOGGER.info("Executing control experiments")

        os.system("./experiments/control.sh")
            
    else: raise ValueError("Experiment must be specified by control, model, dataset, or kernel.")