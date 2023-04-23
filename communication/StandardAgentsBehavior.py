from typing import List

import numpy as np
from agent.CommunicatingAgent import CommunicatingAgent
from ArgumentAgent import ArgumentAgent
from arguments.Argument import Argument
from message.Message import Message
from message.MessagePerformative import MessagePerformative
from preferences.Item import Item
from preferences.Preferences import Preferences


def standard_agent_message_builder(
    agent: ArgumentAgent,
    preferences: Preferences,
    input: Message,
    next_state: MessagePerformative,
) -> Message:
    """Builds a message for the given agent with the given input and next state.

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
        next_state (MessagePerformative): The next state the agent is transitioning.

        Returns:
            Message: The message for the given agent with the given input and next state
    """

    if (
        next_state == MessagePerformative.IDLE
        or next_state == MessagePerformative.FINISHED
    ):
        return None

    if (
        next_state == MessagePerformative.COMMIT
        or next_state == MessagePerformative.ACK
    ):
        content: Argument | Item | str = input.get_content()
        exp = input.get_exp()
        if isinstance(content, Argument):
            item_name = content.__item
        elif isinstance(content, Item):
            item_name = content.get_name()
        elif isinstance(content, str):
            item_name = content
        elif not isinstance(content, str):
            raise Exception("item_name should be a string, an Argument or an Item")

        if exp not in agent.bag:
            agent.bag[exp] = []
        agent.bag[exp].append(item_name)

    if next_state == MessagePerformative.ARGUE:
        item_name = get_item_name(input)
        argument = agent.support_proposal(item_name)
        message = Message(agent.get_name(), input.get_exp(), next_state, argument)
        agent.send_message(message)
        return message

    # if next_state == MessagePerformative.ASK_WHY:
    # item = input.get_content()
    # argument = agent.attack_proposal(item)
    # message = Message(agent.get_name(), input.get_exp(), next_state, argument)
    # agent.send_message(message)
    # return message

    if next_state == MessagePerformative.PROPOSE:
        chosen_agent_name = input.get_dest()
        already_agreed: List[str] = []
        if chosen_agent_name in agent.bag:
            already_agreed = agent.bag[chosen_agent_name]

        item = preferences.most_preferred(
            [x for x in agent.list_items if x.get_name() not in already_agreed],
        )

        chosen_agent: CommunicatingAgent = [
            x for x in agent.model.schedule.agents if x.get_name() == chosen_agent_name
        ][0]
        message = Message(
            agent.get_name(),
            chosen_agent.get_name(),
            next_state,
            item.get_name(),
        )
        agent.send_message(message)
        return message

    if next_state == MessagePerformative.ACCEPT:
        item_name = get_item_name(input)

        exp = input.get_dest()
        dest = input.get_exp()
        message = Message(exp, dest, next_state, item_name)
        agent.send_message(message)
        return message

    exp = input.get_dest()
    dest = input.get_exp()
    content = input.get_content()
    message = Message(exp, dest, next_state, content)
    agent.send_message(message)

    return message


def get_item_name(input: Message) -> str:
    item_name: Argument | Item | str = input.get_content()
    if isinstance(item_name, Argument):
        item_name = item_name.get_item().get_name()
    elif isinstance(item_name, Item):
        item_name = item_name.get_name()
    elif not isinstance(item_name, str):
        raise Exception("item_name should be a string, an Argument or an Item")
    return item_name


def standard_agent_decision_builder(
    agent: ArgumentAgent,
    preferences: Preferences,
    input: Message,
    current_state: MessagePerformative,
    next_states: List[MessagePerformative],
) -> MessagePerformative:
    """This method is responsible for making the agent's decision
    based on its current state, its preferences, and the message received.
    The method returns the next state the agent should transition to.

    Args:
        agent (ArgumentAgent): The argument agent object making the decision.
        preferences (Preferences): preferences object.
        input (Message): The message received by the agent from another agent.
        current_state (MessagePerformative): The current state of the agent.
        next_states (List[MessagePerformative]): A list of the possible next states.

    Returns:
        MessagePerformative: The next state the agent should transition to.
    """
    if current_state == MessagePerformative.IDLE:
        chosen_agent_name = input.get_dest()
        already_agreed: List[str] = []
        if chosen_agent_name in agent.bag:
            already_agreed = agent.bag[chosen_agent_name]

        available_proposals = [
            x for x in agent.list_items if x.get_name() not in already_agreed
        ]
        if len(available_proposals) == 0:
            return MessagePerformative.IDLE

        return np.random.choice(next_states, p=[0.5, 0.5])

    if current_state == MessagePerformative.PROPOSE:
        item_name = get_item_name(input)

        if preferences.most_preferred(agent.list_items).get_name() == item_name:
            return MessagePerformative.ACCEPT

        return MessagePerformative.ASK_WHY

    if current_state == MessagePerformative.ACCEPT:
        return MessagePerformative.COMMIT

    if current_state == MessagePerformative.ARGUE:
        item_name = get_item_name(input)
        item = [x for x in agent.list_items if x.get_name() == item_name][0]
        argument = Argument(False, item)

        list_attacking = argument.list_attacking_proposal(
            item,
            agent.preferences,
        )
        if len(list_attacking) > 0:
            return MessagePerformative.ARGUE

        return MessagePerformative.ACCEPT
