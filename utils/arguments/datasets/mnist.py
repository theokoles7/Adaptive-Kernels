"""MNIST dataset argument definitions."""

from argparse               import ArgumentParser, _SubParsersAction

from utils.arguments.models import  (
                                        add_normal_cnn_parser,
                                        add_resnet_parser,
                                        add_vgg_parser
                                    )

def add_mnist_parser(
    parent_subparser:   _SubParsersAction
) -> None:
    """# Add parser/arguments for MNIST dataset.

    ## Args:
        * parent_subparser  (_SubParsersAction):   Parent's sub-parser.
    """
    # Initialize parser
    _parser_:       ArgumentParser =    parent_subparser.add_parser(
        name =      "mnist",
        help =      """Use MNIST dataset for job process."""
    )
    
    # Initialize sub-parser
    _subparser_:    _SubParsersAction = _parser_.add_subparsers(
        dest =      "model",
        help =      """Model selection."""
    )
    
    # Add MNIST arguments
    _parser_.add_argument(
        "--data-path",
        type =      str,
        default =   "data",
        help =      """Path at which dataset will be downloaded/loaded. Defaults to "./data/"."""
    )
    
    _parser_.add_argument(
        "--batch-size",
        type =      int,
        default =   64,
        help =      """Dataset batch size for training phase. Defaults to 64."""
    )
    
    # Add model parsers
    add_normal_cnn_parser(parent_subparser =    _subparser_)
    add_resnet_parser(parent_subparser =        _subparser_)
    add_vgg_parser(parent_subparser =           _subparser_)