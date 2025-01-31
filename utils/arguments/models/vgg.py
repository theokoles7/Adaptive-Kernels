"""VGG model argument definitions."""

from argparse                   import ArgumentParser, _SubParsersAction

from utils.arguments.kernels    import  (
                                            add_cauchy_parser,
                                            add_gaussian_parser,
                                            add_gumbel_parser,
                                            add_laplace_parser
                                        )

def add_vgg_parser(
    parent_subparser:   _SubParsersAction
) -> None:
    """# Add parser/arguments for VGG model.

    ## Args:
        * parent_subparser  (_SubParsersAction):    Parent's sub-parser.
    """
    # Initialize parser
    _parser_:       ArgumentParser =    parent_subparser.add_parser(
        name =      "vgg",
        help =      """Use VGG model for job process."""
    )
    
    # Add model arguments
    _parser_.add_argument(
        "--learning-rate", "-lr",
        type =      float,
        default =   1e-1,
        help =      """Model's learning rate. Defaults to 0.1."""
    )
    
    _parser_.add_argument(
        "--save-parameters",
        action =    "store_true",
        default =   False,
        help =      """Save model parameters on job completion."""
    )
    
    # Initialize sub-parser
    _subparser_:    _SubParsersAction = _parser_.add_subparsers(
        dest =      "kernel",
        help =      """Kernel selection."""
    )
    
    # Add kernel parser
    add_cauchy_parser(parent_subparser =    _subparser_)
    add_gaussian_parser(parent_subparser =  _subparser_)
    add_gumbel_parser(parent_subparser =    _subparser_)
    add_laplace_parser(parent_subparser =   _subparser_)