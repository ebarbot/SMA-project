import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ArgumentModel import ArgumentModel
from message.MessageService import MessageService
import argparse


def main(n_agents: int):
    print(f"Number of agents: {n_agents}")
    model = ArgumentModel(num_agents=n_agents, verbose=False)
    model.run_n_steps(50)
    df_model = model.datacollector.get_model_vars_dataframe()
    df = pd.DataFrame(dict(df_model.iloc[-1]["Commited"]))
    df = df.applymap(lambda x: [] if not type(x) == list else x)

    total = {}
    df["total"] = np.zeros(len(df))
    for agent in df:
        total[agent] = set()
        for val in df[agent].values:
            total[agent] = total[agent].union(set(val))

        df["total"][agent] = total[agent]

    agreed = df["total"][0]
    for i in df["total"][1:]:
        agreed = agreed.intersection(i)
    print(f"Agreed: {list(agreed)}")

    all_accepted = []
    for i in df["total"]:
        for j in i:
            all_accepted.append(j)

    plt.figure(figsize=(20, 5))
    sns.histplot(all_accepted, bins=len(set(all_accepted)))
    plt.title("Histogram of all accepted items")
    plt.xlabel("Item")
    plt.ylabel("Frequency")

    plt.axhline(y=n_agents, color="r", linestyle="-", label="Number of agents")
    plt.savefig(f"images/histogram_{n_agents}.png")
    del model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the ArgumentModel.")

    parser.add_argument(r"--num_agents", default=3, type=int)

    args, _ = parser.parse_known_args()
    num_agents = args.num_agents
    main(num_agents)
