"""Utilities for logging."""

from utils.globals import *
    
def get_args_banner() -> None:
    """Log important arguments for the record.
    """
    return (
        f"DATASET: {ARGS.dataset} BATCH_SIZE: {ARGS.batch_size} "
        f"| MODEL: {ARGS.model} LR: {ARGS.learning_rate} "
        f"| DISTRO: {ARGS.distribution} {get_distro_params()} KERNEL TYPE: {ARGS.kernel_type}" if ARGS.distribution else ''
    )

def get_distro_params() -> str:
    """Provide distribution parameters.

    Returns:
        str: Distribution parameters
    """
    if ARGS.distribution == 'poisson': return f"RATE: {ARGS.rate}"
    return f"LOC: {ARGS.location} SCALE: {ARGS.scale}"