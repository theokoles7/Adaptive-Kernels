"""Experiment argument definitions."""

from argparse                   import ArgumentParser, _SubParsersAction

def add_experiment_parser(
    parent_subparser:   _SubParsersAction
) -> None:
    """# Add parser/arguments for running an experiment (job series).

    ## Args:
        * parent_subparser  (_SubParsersAction):    Parent's sub-parser.
    """
    # Inialize parser
    _parser_:       ArgumentParser =    parent_subparser.add_parser(
        name =      "run-experiment",
        help =      """Execute an experiment process (job series)."""
    )
    
    # Add experiment arguments
    _parser_.add_argument(
        "datasets",
        type =      str,
        nargs =     "+",
        choices =   ["cifar10", "cifar100", "imagenet", "mnist"],
        help =      """Dataset(s) on which experiments will be executed."""
    )
    
    _parser_.add_argument(
        "models",
        type =      str,
        nargs =     "+",
        choices =   ["normal-cnn", "resnet", "vgg"],
        help =      """Model(s) on which experiments will be executed."""
    )
    
    _parser_.add_argument(
        "kernels",
        type =      str,
        nargs =     "+",
        choices =   ["cauchy", "gaussian", "gumbel", "laplace"],
        help =      """Kernel(s) on which experiments will be executed. Equivalent of experiment(s) 
                    with no kernel will automatically be executed for comparison."""
    )