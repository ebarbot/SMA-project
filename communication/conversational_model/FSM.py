#!/usr/bin/env python3
from abc import ABC, abstractmethod
import json
from mailbox import Message
from pathlib import Path
from typing import Any
import networkx as nx
import matplotlib.pyplot as plt
from networkx.readwrite import json_graph
import pprint

import sys
import os
print(os.getcwd())
sys.path.append(os.getcwd())
from message.MessagePerformative import MessagePerformative  # nopep8
from preferences.Preferences import Preferences  # nopep8


class FiniteStateMachine(ABC):

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def step(self, input: Any) -> MessagePerformative:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def is_finished(self) -> bool:
        pass


class BasicFSM(FiniteStateMachine):
    def __init__(self, path: str = '.', filename: str = 'conversational_graph.json', verbose: bool = True) -> None:

        filename = Path(path) / Path(filename)
        print('filename: {}'.format(filename))
        with open(filename, 'r') as json_file:
            graph_data = json.load(json_file)

        print('Generating graph from file: {}'.format(filename))
        self.graph = json_graph.node_link_graph(graph_data)

        if verbose:
            print('-' * 80)
            print('Graph data: ')
            pprint.pprint(graph_data)
            print('-' * 80)

            plt.figure(1, figsize=(5, 5))
            nx.draw(self.graph, with_labels=True)
            plt.show()

    def step(self, input: Message, preferences: Preferences) -> MessagePerformative:
        super().step(input)

        next_states = self.graph.successors(input.performative.name)

    def reset(self) -> None:
        return super().reset()

    def is_finished(self) -> bool:
        return super().is_finished()


if __name__ == '__main__':

    fsm = FiniteStateMachine()
