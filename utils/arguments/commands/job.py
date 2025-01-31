"""Job argument definitions."""

from argparse                   import ArgumentParser, _SubParsersAction

from utils.arguments.datasets   import  (
                                            add_cifar10_parser,
                                            add_cifar100_parser,
                                            add_imagenet_parser,
                                            add_mnist_parser
                                        )

def add_job_parser(
    parent_subparser:   _SubParsersAction
) -> None:
    """# Add parser/arguments for running a job.

    ## Args:
        * parent_subparser  (_SubParsersAction):    Parent's sub-parser.
    """
    # Initialize parser
    _parser_:       ArgumentParser =    parent_subparser.add_parser(
        name =      "run-job",
        help =      """Execute an individual job process."""
    )
    
    # Initialize sub-parser
    _subparser_:    _SubParsersAction = _parser_.add_subparsers(
        dest =      "dataset",
        help =      """Dataset selection."""
    )
    
    # Add job argument(s)
    _parser_.add_argument(
        "--epochs",
        type =      int,
        default =   200,
        help =      """Number of epochs for which training phase of job will execute. Defaults to 
                    200."""
    )
    
    # Add dataset parsers
    add_cifar10_parser(parent_subparser =   _subparser_)
    add_cifar100_parser(parent_subparser =  _subparser_)
    add_imagenet_parser(parent_subparser =  _subparser_)
    add_mnist_parser(parent_subparser =     _subparser_)