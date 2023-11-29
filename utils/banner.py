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
    f"\n| {'MODEL:':<14}{ARGS.model:>15} | {'LR:':<14}{ARGS.learning_rate:>14} |" +
    (f"\n| {'DISTRO:':<14}{ARGS.distribution:>15} | {'KERNEL TYPE:':<14}{ARGS.kernel_type:>14} |" if ARGS.distribution else '') +
    f"\n| {'DATASET:':<14}{ARGS.dataset:>15} | {'BATCH SIZE:':<14}{ARGS.batch_size:>14} |" +
     "\n+=========================================================================+"
)