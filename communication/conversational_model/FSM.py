#!/usr/bin/env python3
import inspect
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
    def has_finished(self) -> bool:
        pass


class Turn:
    Me: int = 0
    Other: int = 1


class FiniteStateMachine(FiniteStateMachineBase):
    def __init__(self, agent_a: str, agent_b: str, path: str = '.', filename: str = 'conversational_graph.json', verbose: int = 0) -> None:

        filename = Path(path) / Path(filename)
        with open(filename, 'r') as json_file:
            graph_data: ConversationalGraph = json.load(json_file)

        self.graph = json_graph.node_link_graph(graph_data)

        self.__sanity_check(graph_data)

        initial_states = [
            node['id'] for node in graph_data['nodes'] if 'initial' in node]

        final_states = [
            MessagePerformative[node['id']] for node in graph_data['nodes'] if 'final' in node]

        # TODO: Could there be more than one initial state?
        assert len(initial_states) == 1, 'More than one initial state'

        self.initial_state: MessagePerformative = MessagePerformative[initial_states[0]]
        self.final_states: List[MessagePerformative] = final_states
        self.current_state: MessagePerformative = self.initial_state
        self.turn: Turn = Turn.Me
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.verbose = verbose

        if self.verbose == 2:
            print('-' * 80)
            print('Graph data: ')
            pprint.pprint(graph_data)
            print('-' * 80)

            plt.figure(1, figsize=(5, 5))
            nx.draw(self.graph, with_labels=True)
            plt.show()

    def __sanity_check(self, graph_data):
        attributes = inspect.getmembers(
            MessagePerformative, lambda a: not(inspect.isroutine(a)))

        performatives = sorted(set([a[0] for a in attributes if not(
            (a[0].startswith('__') and a[0].endswith('__')) or 'name' in a[0] or 'value' in a[0])]))

        nodes = sorted([node['id'] for node in graph_data['nodes']])
        assert nodes == performatives, f'Performatives in graph do not match those in MessagePerformative: {[x for x in nodes if x not in performatives]} - {[x for x in performatives if x not in nodes]}'

    def step(self, input: Message = None, preferences: Preferences = None):
        if preferences:
            self.turn = Turn.Other
            msg = self.__my_step(input, preferences)
            return msg

        self.__infer_step(input)
        self.turn = Turn.Me

    def __infer_step(self, input: Message) -> None:
        next_state = input.get_performative()
        self.current_state = next_state

    def __my_step(self, input: Message, preferences: Preferences) -> Message:
        super().step(input)
        if self.verbose >= 1:
            print('\n[FiniteStateMachine]: Agents: {} {}'.format(
                self.agent_a, self.agent_b), end=' ')

        next_states = [MessagePerformative[x] for x in list(
            self.graph.successors(self.current_state.name))]

        if len(next_states) > 1:
            next_state = preferences.decide(
                input, self.current_state, next_states)
            assert next_state in next_states, f'Next state was not defined in the graph: {next_state} not in {next_states}'
        else:
            next_state = next_states[0]

        if self.verbose >= 1:
            print('Current state: {}'.format(self.current_state),
                  'Next state: {}'.format(next_state))
        msg = preferences.build_message(input, next_state)

        self.current_state = next_state

        return msg

    def reset(self) -> None:
        self.current_state = self.initial_state
        self.turn = Turn.Me

    def has_finished(self) -> bool:
        return self.current_state in self.final_states and self.turn == Turn.Other

    def is_start(self) -> bool:
        return self.current_state == self.initial_state


if __name__ == '__main__':

    fsm = FiniteStateMachine()
