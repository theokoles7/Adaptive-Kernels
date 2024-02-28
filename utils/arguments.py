"""Command line arguments utilities."""

import  argparse

# Initialize parser
parser =            argparse.ArgumentParser(
    "hilo",
    description =   (
        "Fork of KernelSet, intended for studying "
        "the efficacy of models on high and low "
        "resolution images, using various "
        "probability distributions as kernels."
    )
)

# Initialize subparsers
sub_parsers =       parser.add_subparsers(
    dest =          "cmd",
    description =   "Command being executed"
)

###################################################################################################
# BEGIN ARGUMENTS                                                                                 #
###################################################################################################

# UNIVERSAL =======================================================================================

# LOGGING ----------------------------------------------------------------
logging =           parser.add_argument_group("Logging")

logging.add_argument(
    "--logging_path",
    type =          str,
    default =       "logs",
    help =          "Directory at which log files will be written. Defaults to \'./logs/\'."
)

logging.add_argument(
    "--logging_level",
    type =          str,
    choices =       ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    default =       "INFO",
    help =          "Minimum logging level (DEBUG < INFO < WARNING < ERROR < CRITICAL). Defaults to \'INFO\'."
)

# OUTPUT -----------------------------------------------------------------
output =            parser.add_argument_group("Output")

output.add_argument(
    "--output_path",
    type =          str,
    default =       "output",
    help =          "Directory at which output/results files will be written. Defaults to \'./output/\'."
)

# AUXILIARY FUNCTIONS =============================================================================

# INITIALIZE RESULTS FILE ------------------------------------------------
sub_parsers.add_parser(
    "init-results",
    description =   "Initialize results CSV file."
)

# EXPERIMENTS ------------------------------------------------------------

experiments =       sub_parsers.add_parser(
    "run-experiment",
    description =   "Run predetermined series of jobs for a specific model, dataset, or kernel"
)

experiments.add_argument(
    "--control",
    action =        "store_true",
    help =          "Run control experiments (Without distribution kernels)"
)

experiments.add_argument(
    "--by_model",
    type =          str,
    choices =       ["normal", "resnet", "vgg"],
    help =          "Run model jobs with all variations of kernels and datasets"
)

experiments.add_argument(
    "--by_kernel",
    type =          str,
    choices =       ["cauchy", "gaussian", "gumbel", "laplace", "poisson"],
    help =          "Run kernel jobs with all variations of models and datasets"
)

experiments.add_argument(
    "--by_dataset",
    type =          str,
    choices =       ["cifar10", "cifar100", "imagenet", "mnist"],
    help =          "Run dataset jobs with all variations of kernels and models"
)

# JOBS -------------------------------------------------------------------

jobs = sub_parsers.add_parser(
    "run-job",
    description =   "Run job with choice of model, dataset, and kernel"
)

jobs.add_argument(
    "--epochs",
    type =          int,
    default =       200,
    help =          "Number of training/validation epochs to run in job"
)

# MODEL -----------------------------------------
model =             jobs.add_argument_group("Model")

model.add_argument(
    "model",
    type =          str,
    choices =       ["normal", "resnet", "vgg"],
    help =          "Choice of CNN model"
)

model.add_argument(
    "--learning_rate", "-lr",
    type =          float,
    default =       1e-1,
    help =          "Optimizer learning rate. Defaults to 0.1"
)

model.add_argument(
    "--save_params",
    action =        "store_true",
    help =          "Save model parameters on job completion"
)

# DATASET ---------------------------------------
dataset =           jobs.add_argument_group("Dataset")

dataset.add_argument(
    "dataset",
    type =          str,
    choices =       ["cifar10", "cifar100", "imagenet", "mnist"],
    help =          "Choice of dataset"
)

dataset.add_argument(
    "--dataset_path",
    type =          str,
    default =       "data",
    help =          "Directory at which datasets will be downloaded/pulled. Defaults to \'./data/\'."
)

dataset.add_argument(
    "--batch_size",
    type =          int,
    default =       64,
    help =          "Dataset batch size. Defaults to 64."
)

# KERNEL ----------------------------------------
kernel =            jobs.add_argument_group("Kernel")

kernels =           jobs.add_subparsers(
    dest =          "distribution",
    description =   "Choice of kernel distribution"
)

kernel.add_argument(
    "--kernel_size",
    type =          int,
    default =       3,
    help =          "Kernel size (square). Defaults to 3."
)

kernel.add_argument(
    "--kernel_type",
    type =          int,
    choices =       [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    default =       13,
    help =          "Kernel configuration type. Defaults to 13."
)

# CAUCHY _______________
cauchy = kernels.add_parser(
    "cauchy",
    description =   "Cauchy kernel distribution"
)

cauchy.add_argument(
    "--location",
    type =          float,
    default =       0.0,
    help =          "Cauchy distribution location parameter (Chi)"
)

cauchy.add_argument(
    "--scale",
    type =          float,
    default =       1.0,
    help =          "Cauchy distribution scale parameter (Gamma)"
)

# GAUSSIAN _____________
gaussian = kernels.add_parser(
    "gaussian",
    description =   "Gaussian kernel distribution"
)

gaussian.add_argument(
    "--location",
    type =          float,
    default =       0,
    help =          "Gaussian distribution location parameter (Mu)"
)

gaussian.add_argument(
    "--scale",
    type =          float,
    default =       1.0,
    help =          "Gaussian distribution scale parameter (Sigma)"
)

# GUMBEL _______________
gumbel = kernels.add_parser(
    "gumbel",
    description =   "Gumbel kernel distribution"
)

gumbel.add_argument(
    "--location",
    type =          float,
    default =       0.0,
    help =          "Gumbel distribution location parameter (Mu)"
)

gumbel.add_argument(
    "--scale",
    type =          float,
    default =       1.0,
    help =          "Gumbel distribution scale parameter (Beta)"
)

# LAPLACE ______________
laplace = kernels.add_parser(
    "laplace",
    description =   "Laplace kernel distribution"
)

laplace.add_argument(
    "--location",
    type =          float,
    default =       0.0,
    help =          "Laplace distribution location parameter (Mu)"
)

laplace.add_argument(
    "--scale",
    type =          float,
    default =       1.0,
    help =          "Laplace distribution scale parameter (Sigma)"
)

# POISSON ______________
poisson = kernels.add_parser(
    "poisson",
    description =   "Poisson kernel distribution"
)

poisson.add_argument(
    "--rate",
    type =          float,
    default =       1.0,
    help =          "Poisson distribution rate parameter (Lambda)"
)

poisson.add_argument(
    "--scale",
    type =          float,
    default =       1.0,
    help =          "Not used; Ignore this option"
)

###################################################################################################
# END ARGUMENTS                                                                                   #
###################################################################################################

# Parse command line arguments
ARGS =              parser.parse_args()