"""Laplace kernel argument definitions."""

from argparse   import ArgumentParser, _SubParsersAction

def add_laplace_parser(
    parent_subparser:   _SubParsersAction
) -> None:
    """# Add parser/arguments for Laplace kernel.

    ## Args:
        * parent_subparser  (_SubParsersAction):    Parent's sub-parser.
    """
    # Initialize parser
    _parser_:       ArgumentParser =    parent_subparser.add_parser(
        name =      "laplace",
        help =      """Use Laplace kernel for job process."""
    )
    
    # Add Laplace kernel arguments
    _parser_.add_argument(
        "--location", "-mu",
        type =      float,
        default =   0.0,
        help =      """Location (Mu) parameter. Defaults to 0."""
    )
    
    _parser_.add_argument(
        "--scale", "-beta",
        type =      float,
        default =   1.0,
        help =      """Scale (Beta) parameter. Defaults to 1."""
    )
    
    _parser_.add_argument(
        "--size",
        type =      int,
        default =   3,
        help =      """Kernel size (square). Defaults to 3."""
    )
    
    _parser_.add_argument(
        "--kernel-group",
        type =      int,
        default =   13,
        help =      """Kernel configuration type. Defaults to 13."""
    )