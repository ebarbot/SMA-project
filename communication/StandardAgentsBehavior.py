#!/usr/bin/env python3

import copy

from agent.CommunicatingAgent import CommunicatingAgent
from preferences.Preferences import Preferences
from preferences.Item import Item
from message.Message import Message
from message.MessagePerformative import MessagePerformative
from arguments.Argument import Argument
from ArgumentAgent import ArgumentAgent

from typing import List, Union
import numpy as np


def standard_agent_message_builder(agent: ArgumentAgent, preferences: Preferences, input: Message, next_state: MessagePerformative) -> Message:
    """ Builds a message for the given agent with the given input and next state.

        To build a message for a performative <PERFORMATIVE>,
        one simply needs to create an IF statement for the given performative
        Practical example:

            >>> if next_state == MessagePerformative.ARGUE:
            ...    (do something on COMMIT)
            ...    message = Message(agent, next_state, argument)
            >>> return message 

        Args:
            agent (ArgumentAgent): The agent for which the message is being built.
            preferences (Preferences): The preferences of the agent.
            input (Message): The input message received by the agent.
            next_state (MessagePerformative): The next state to which the agent is transitioning.

            Returns:
                Message: The message built for the given agent with the given input and next state.
    """

    if next_state == MessagePerformative.IDLE or next_state == MessagePerformative.FINISHED:
        return None

    if next_state == MessagePerformative.COMMIT:
        item_name: Union[Argument, Item, str] = input.get_content()
        if isinstance(item_name, Argument):
            item_name = item_name.item
        elif isinstance(item_name, Item):
            item_name = item_name.get_name()
        elif not isinstance(item_name, str):
            raise Exception(
                'item_name should be a string, an Argument or an Item')

        item = [x for x in agent.bag if x.get_name() == item_name][0]
        agent.bag.remove(item)

    if next_state == MessagePerformative.ARGUE:
        argument: Argument = input.get_content()
        argument = agent.support_proposal(argument.item)
        message = Message(agent.get_name(),
                          input.get_exp(), next_state, argument)
        agent.send_message(message)
        return message

    if next_state == MessagePerformative.ASK_WHY:
        item = input.get_content()
        argument = agent.attack_proposal(item)
        message = Message(agent.get_name(),
                          input.get_exp(), next_state, argument)
        agent.send_message(message)
        return message

    if next_state == MessagePerformative.PROPOSE:
        item = preferences.most_preferred(agent.bag)
        chosen_agent_name = input.get_dest()
        chosen_agent: CommunicatingAgent = [
            x for x in agent.model.schedule.agents if x.get_name() == chosen_agent_name][0]
        message = Message(agent.get_name(),
                          chosen_agent.get_name(), next_state, item.get_name())
        agent.send_message(message)
        return message

    if next_state == MessagePerformative.ACK:
        item_name: Union[Argument, Item, str] = input.get_content()
        if isinstance(item_name, Argument):
            item_name = item_name.item
        elif isinstance(item_name, Item):
            item_name = item_name.get_name()
        elif not isinstance(item_name, str):
            raise Exception(
                'item_name should be a string, an Argument or an Item')

        item = [x for x in agent.list_items if x.get_name() == item_name][0]
        agent.bag.append(copy.deepcopy(item))

    exp = input.get_dest()
    dest = input.get_exp()
    content = input.get_content()
    message = Message(exp, dest, next_state, content)
    agent.send_message(message)

    return message


def standard_agent_decision_builder(agent: ArgumentAgent, preferences: Preferences, input: Message, current_state: MessagePerformative, next_states: List[MessagePerformative]) -> MessagePerformative:
    """ This method is responsible for making the agent's decision 
    based on its current state, its preferences, and the message received. 
    The method returns the next state the agent should transition to.

    Args:
        agent (ArgumentAgent): The argument agent object making the decision.
        preferences (Preferences): The preferences object that contains the agent's preferences.
        input (Message): The message received by the agent from another agent.
        current_state (MessagePerformative): The current state of the agent.
        next_states (List[MessagePerformative]): A list of the possible next states for the agent to transition to.

    Returns:
        MessagePerformative: The next state (MessagePerformative) the agent should transition to.
    """
    if current_state == MessagePerformative.IDLE:
        if len(agent.bag) > 0:
            return np.random.choice(next_states, p=[0.5, 0.5])
        return MessagePerformative.IDLE

    if current_state == MessagePerformative.PROPOSE:
        item_name = input.get_content()
        if item_name in [x.get_name() for x in agent.bag]:
            return MessagePerformative.REJECT

        item = [x for x in agent.list_items if x.get_name() == item_name][0]
        list_attacking = Argument(True, item).list_attacking_proposal(
            item, agent.preferences)

        if len(list_attacking) > 0:
            return MessagePerformative.ASK_WHY

        return MessagePerformative.ACCEPT

    if current_state == MessagePerformative.ACCEPT:
        item_name: Union[Argument, Item, str] = input.get_content()
        item = [x for x in agent.bag if x.get_name() == item_name]
        if len(item) == 0:
            return MessagePerformative.REJECT
        return MessagePerformative.COMMIT

    if current_state == MessagePerformative.ARGUE:
        return MessagePerformative.ACCEPT
