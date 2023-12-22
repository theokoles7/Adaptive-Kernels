"""Initialize results CSV file."""

import csv

from utils  import ARGS

def init_results() -> None:
    """Initialize results CSV file."""

    # Create/open CSV file
    with open(f"{ARGS.output_path}/results.csv", 'w', newline='') as file_out:

        # Initialize writer
        writer = csv.writer(file_out)

        # Write header row
        writer.writerow(["MODEL", "DATASET", "DISTRIBUTION", "KERNEL TYPE", "BEST ACCURACY", "EPOCH", "TEST ACCURACY"])

        # Write rows
        for model in ["normal", "resnet", "vgg"]:
            for dataset in ["cifar10", "cifar100", "imagenet", "mnist"]:
                for distro in ["none", "cauchy", "gaussian", "gumbel", "laplace", "poisson"]:

                    # If no distribution...
                    if distro == "none":
                        # Write one row, with no kernel type
                        writer.writerow([model, dataset, distro, "--", "--", "--", "--"])

                    # Otherwise, a distro must be used, so...
                    else:
                        # Write a row for each kernel type
                        for i in range(1, 15):
                            writer.writerow([model, dataset, distro, i, "--", "--", "--"])