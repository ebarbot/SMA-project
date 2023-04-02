#!/usr/bin/env python3
# Description: This file contains the ArgumentModel class which is the main class of the argumentation model.

import copy
from mesa import Model
from mesa.time import RandomActivation
from ArgumentAgent import ArgumentAgent
from StandardAgentsBehavior import standard_agent_decision_builder, standard_agent_message_builder

from preferences.ItemFactory import ItemCreator_CSV
from message.MessageService import MessageService

import numpy as np


class ArgumentModel(Model):
    """ 
        The ArgumentModel class is a model that simulates a group of agents who participate in an argument. The model is inherited from the base Model class.

        The class constructor takes one argument:
        1. num_agents - an integer that specifies the number of agents in the simulation. Default value is 2.

    """

    def __init__(self, num_agents: int = 2):
        """
        Initializes a new ArgumentModel object.

        Args:
            num_agents (int): The number of agents in the simulation. Default value is 2.

        Attributes:
            schedule (RandomActivation): A scheduler that runs the agents in a random order.
            __messages_service (MessageService): A service that manages message passing between agents.
            current_id (int): A counter that keeps track of the current agent id.

        Notes:
            The ArgumentModel assumes that an ItemCreator_CSV class has been defined elsewhere that has a create() method
            that returns a tuple of items_list and map_item_criterion.
        """

        self.schedule = RandomActivation(self)
        self.__messages_service = MessageService(self.schedule)

        itemCreator = ItemCreator_CSV()
        items_list, map_item_criterion = itemCreator.create()

        self.current_id = 0
        for i in range(num_agents):
            new_agent = self.__create_agent()
            new_agent.set_bag(list(np.array(copy.deepcopy(items_list))[np.random.randint(
                0, len(items_list), size=np.random.randint(len(items_list), size=1))]))
            new_agent.generate_preferences(
                copy.deepcopy(items_list), copy.deepcopy(map_item_criterion), verbose=2)
            self.schedule.add(new_agent)

        self.running = True

    def __create_agent(self) -> ArgumentAgent:
        # Creates a new agent and returns it.
        return ArgumentAgent(self.next_id(), self, "Agent " + str(self.current_id), standard_agent_decision_builder, standard_agent_message_builder)

    def step(self):
        # Runs one step of the simulation.
        self.__messages_service.dispatch_messages()
        self.schedule.step()

    def run_n_steps(self, n: int):
        # Runs n steps of the simulation.
        for _ in range(n):
            print('-'*80)
            self.step()
