#!/usr/bin/env python3
from ArgumentModel import ArgumentModel

if __name__ == "__main__":
    model = ArgumentModel(num_agents=5)

    model.run_n_steps(20)
