"""Logging banners."""

from utils.arguments import ARGS

def get_distro_params() -> str:
    """Provide distribution parameters.

    Returns:
        str: Distribution parameters
    """
    if ARGS.distribution == 'poisson': return f"RATE: {ARGS.rate}"
    return f"LOC: {ARGS.location} SCALE: {ARGS.scale}"

lab_banner = open('utils/lab_banner.txt', 'r').read()

arg_banner = (
    f"\n| {'MODEL:':<17}{ARGS.model:>17} | {'LR:':<17}{ARGS.learning_rate:>17} |" +
    (f"\n| {'DISTRO:':<17}{ARGS.distribution:>17} | {'KERNEL TYPE:':<17}{ARGS.kernel_type:>17} |" if ARGS.distribution else '') +
    f"\n| {'DATASET:':<17}{ARGS.dataset:>17} | {'BATCH SIZE:':<17}{ARGS.batch_size:>17} |" +
     "\n+=========================================================================+"
)