#!/usr/bin/env python3
from typing import Dict, List, Optional, TypedDict
from abc import ABC, abstractmethod
import json
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
from message.Message import Message  # nopep8
from message.MessagePerformative import MessagePerformative  # nopep8
from preferences.Preferences import Preferences  # nopep8


class GraphNode(TypedDict):
    id: MessagePerformative
    initial: Optional[bool]
    final: Optional[bool]


class Links(TypedDict):
    source: MessagePerformative
    target: MessagePerformative


class ConversationalGraph(TypedDict):
    directed: bool
    multigraph: bool
    graph: Dict
    nodes: List[GraphNode]
    links: List[Links]


class FiniteStateMachineBase(ABC):

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


class Turn:
    Me: int = 0
    Other: int = 1


class FiniteStateMachine(FiniteStateMachineBase):
    def __init__(self, path: str = '.', filename: str = 'conversational_graph.json', verbose: bool = True) -> None:

        filename = Path(path) / Path(filename)
        print('filename: {}'.format(filename))
        with open(filename, 'r') as json_file:
            graph_data: ConversationalGraph = json.load(json_file)

        print('Generating graph from file: {}'.format(filename))
        self.graph = json_graph.node_link_graph(graph_data)

        initial_states = [
            node['id'] for node in graph_data['nodes'] if 'initial' in node]

        final_states = [
            node for node in graph_data['nodes'] if 'final' in node]

        # TODO: Could there be more than one initial state?
        assert len(initial_states) == 1, 'More than one initial state'

        self.initial_state: MessagePerformative = initial_states[0]
        self.final_state: MessagePerformative = final_states[0]
        self.current_state: MessagePerformative = self.initial_state
        self.turn: Turn = Turn.Me

        if verbose:
            print('-' * 80)
            print('Graph data: ')
            pprint.pprint(graph_data)
            print('-' * 80)

            plt.figure(1, figsize=(5, 5))
            nx.draw(self.graph, with_labels=True)
            plt.show()

    def step(self, input: Message = None, preferences: Preferences = None):
        if preferences:
            self.turn = Turn.Other
            msg = self.__my_step(input, preferences)
            return msg

        self.__infer_step(input)
        self.turn = Turn.Me

    def __build_message(self, input: Message, next_state: MessagePerformative) -> Message:
        exp = input.get_dest()
        dest = input.get_exp()
        item = input.get_content()
        return Message(exp, dest, next_state, item)

    def __infer_step(self, input: Message) -> None:
        next_state = input.get_performative()
        self.current_state = next_state

    def __my_step(self, input: Message, preferences: Preferences) -> Message:
        super().step(input)

        if self.is_finished():
            self.reset()

        print('Current state: {}'.format(self.current_state))
        next_states = list(self.graph.successors(self.current_state))

        if len(next_states) > 0:
            next_state = preferences.decide(
                input, self.current_state, next_states)
            assert next_state in next_states, 'Next state was not defined in the graph'
        else:
            next_state = next_states[0]

        msg = preferences.build_message(input, next_state)

        self.current_state = next_state

        return msg

    def reset(self) -> None:
        self.current_state = self.initial_state

    def is_finished(self) -> bool:
        return self.current_state == self.final_state

    def is_start(self) -> bool:
        return self.current_state == self.initial_state


if __name__ == '__main__':

    fsm = FiniteStateMachine()
