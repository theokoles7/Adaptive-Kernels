"""Cauchy kernel argument definitions."""

from argparse   import ArgumentParser, _SubParsersAction

def add_cauchy_parser(
    parent_subparser:   _SubParsersAction
) -> None:
    """# Add parser/arguments for Cauchy kernel.

    ## Args:
        * parent_subparser  (_SubParsersAction):    Parent's sub-parser.
    """
    # Initialize parser
    _parser_:       ArgumentParser =    parent_subparser.add_parser(
        name =      "cauchy",
        help =      """Use Cauchy kernel for job process."""
    )
    
    # Add Cauchy kernel arguments
    _parser_.add_argument(
        "--location", "-chi",
        type =      float,
        default =   0.0,
        help =      """Location (Chi) parameter. Defaults to 0."""
    )
    
    _parser_.add_argument(
        "--scale", "-gamma",
        type =      float,
        default =   1.0,
        help =      """Scale (Gamma) parameter. Defaults to 1."""
    )
    
    _parser_.add_argument(
        "--size",
        type =      int,
        default =   3,
        help =      """Kernel size (square). Defaults to 3."""
    )
    
    _parser_.add_argument(
        "--type",
        type =      int,
        default =   13,
        help =      """Kernel configuration type. Defaults to 13."""
    )