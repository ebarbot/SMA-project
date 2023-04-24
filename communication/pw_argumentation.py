import argparse

from ArgumentModel import ArgumentModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the ArgumentModel.")

    # add verbose tag
    parser.add_argument(
        r"--verbose",
        action="store_true",
        help="Prints the state of the model at each step.",
    )
    parser.add_argument(
        r"--num_agents",
        default=3,
        type=int,
        help="Number of agents in the model.",
    )
    parser.add_argument(
        r"--num_iter",
        default=50,
        type=int,
        help="Number of iterations to run the model.",
    )
    args, _ = parser.parse_known_args()

    verbose = args.verbose
    num_agents = args.num_agents
    num_iter = args.num_iter

    model = ArgumentModel(num_agents=num_agents, verbose=verbose)

    model.run_n_steps(num_iter)
