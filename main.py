"""Drive operations of HI/LO Machine Learning application."""

from globals import *

def config_banner() -> None:
    LOGGER.info(
        f"DATASET: {ARGS.dataset} BATCH_SIZE: {ARGS.batch_size} | MODEL: {ARGS.model} LR: {ARGS.learning_rate} | DISTRO: {ARGS.distribution} {distro_params(ARGS.distribution)}"
    )

def distro_params(distro: str) -> str:
    """Provide distribution parameters.

    Args:
        distro (str): Distribution argument

    Returns:
        str: Distributino parameters
    """
    if distro == 'poisson': return f"RATE: {ARGS.rate}"
    return f"LOC: {ARGS.location} SCALE: {ARGS.scale}"

if __name__ == '__main__':
    # Log run time configuration
    config_banner()