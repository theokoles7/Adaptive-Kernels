"""Drive application."""

from arguments import get_args
from solver_cbs import CBSSolver

def main():
    """Initiate main process.
    """
    # Parse arguments.
    args = get_args()

    # Initialize model.
    solver = CBSSolver(args)

    # Initiate operations.
    solver.solve()

    # If argument was passed...
    if args.save_model:

        # Save model parameters.
        solver.save_model()

if __name__ == '__main__':
    main()
