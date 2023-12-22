"""Run predetermined experiments."""

import os

from utils  import ARGS, LOGGER

def run_experiment() -> None:
    """Execute bash script outlining predetermined experiment."""

    # Determine if experiment is being run by model, dataset, or kernel
    if ARGS.by_model:
        LOGGER.info(f"Executing experiment by model: {ARGS.by_model}")

        match ARGS.by_model:

            case "normal":
                os.system("./experiments/by_model/normal.sh")

            case "resnet":
                os.system("./experiments/by_model/resnet.sh")

            case "vgg":
                os.system("./experiments/by_model/vgg.sh")

            case _:
                raise NotImplementedError(f"Experiment by model not supported for {ARGS.by_model}")

    elif ARGS.by_kernel:
        LOGGER.info(f"Executing experiment by kernel: {ARGS.by_kernel}")

        match ARGS.by_kernel:

            case "cauchy":
                os.system("./experiments/by_kernel/cauchy.sh")

            case "gaussian":
                os.system("./experiments/by_kernel/gaussian.sh")

            case "gumbel":
                os.system("./experiments/by_kernel/gumbel.sh")

            case "laplace":
                os.system("./experiments/by_kernel/laplace.sh")

            case "poisson":
                os.system("./experiments/by_kernel/poisson.sh")

            case _:
                raise NotImplementedError(f"Experiment by kernel not supported for {ARGS.by_kernel}")

    elif ARGS.by_dataset:
        LOGGER.info(f"Executing experiment by dataset: {ARGS.by_dataset}")

        match ARGS.by_dataset:

            case "cifar10":
                os.system("./experiments/by_dataset/cifar10.sh")

            case "cifar100":
                os.system("./experiments/by_dataset/cifar100.sh")

            case "imagenet":
                os.system("./experiments/by_dataset/imagenet.sh")

            case "mnist":
                os.system("./experiments/by_dataset/mnist.sh")

            case _:
                raise NotImplementedError(f"Experiment by dataset not supported for {ARGS.by_dataset}")
            
    elif ARGS.control:
        LOGGER.info("Executing control experiments")

        os.system("./experiments/control.sh")
            
    else: raise ValueError("Experiment must be specified by control, model, dataset, or kernel.")