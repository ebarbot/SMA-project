from ArgumentModel import ArgumentModel

if __name__ == "__main__":
    model = ArgumentModel(num_agents=2)

    model.run_n_steps(20)
