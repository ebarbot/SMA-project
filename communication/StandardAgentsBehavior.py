from typing import List

import numpy as np
from agent.CommunicatingAgent import CommunicatingAgent
from ArgumentAgent import ArgumentAgent, Argumentation
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
        item_name = get_item_name(input)

        if exp not in agent.agreed_items:
            agent.agreed_items[exp] = []
        if item_name not in agent.agreed_items[exp]:
            agent.agreed_items[exp].append(item_name)

    if next_state == MessagePerformative.ARGUE:
        item_name = get_item_name(input)
        proposed_argument = input.get_content()
        argument = agent.parse_argument(proposed_argument)
        if input.get_exp() not in agent.argumentations:
            agent.argumentations[input.get_exp()] = Argumentation(
                agent.get_name(),
                input.get_exp(),
            )

        agent.argumentations[input.get_exp()].add_argument(argument)

        argument.set_parent(proposed_argument)
        message = Message(agent.get_name(), input.get_exp(), next_state, argument)
        agent.send_message(message)
        return message

    if next_state == MessagePerformative.BECAUSE:
        item = input.get_content()
        if input.get_exp() not in agent.argumentations:
            agent.argumentations[input.get_exp()] = Argumentation(
                agent.get_name(),
                input.get_exp(),
            )
        argument = agent.support_proposal(item, input.get_exp())
        agent.argumentations[input.get_exp()].add_argument(argument)
        argument.set_parent(None)
        message = Message(agent.get_name(), input.get_exp(), next_state, argument)
        agent.send_message(message)
        return message

    if next_state == MessagePerformative.PROPOSE:
        chosen_agent_name = input.get_dest()
        already_agreed: List[str] = []
        if chosen_agent_name in agent.agreed_items:
            already_agreed = agent.agreed_items[chosen_agent_name]

        already_proposed = []
        if chosen_agent_name in agent.proposed_items:
            already_proposed = agent.proposed_items[chosen_agent_name]

        available_proposals = [
            x
            for x in agent.list_items
            if x.get_name() not in already_agreed + already_proposed
        ]
        item = preferences.most_preferred(available_proposals)
        if chosen_agent_name not in agent.proposed_items:
            agent.proposed_items[chosen_agent_name] = []
        agent.proposed_items[chosen_agent_name].append(item.get_name())

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

    if next_state == MessagePerformative.QUERY_REF:
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
        already_proposed: List[str] = []
        if chosen_agent_name in agent.agreed_items:
            already_agreed = agent.agreed_items[chosen_agent_name]
        if chosen_agent_name in agent.proposed_items:
            already_proposed = agent.proposed_items[chosen_agent_name]

        available_proposals = [
            x
            for x in agent.list_items
            if x.get_name() not in already_agreed + already_proposed
        ]
        if len(available_proposals) == 0:
            return MessagePerformative.IDLE

        item = agent.preferences.most_preferred(available_proposals)
        argument = agent.support_proposal(item.get_name(), input.get_dest())
        if not argument:
            return MessagePerformative.IDLE

        return np.random.choice(next_states, p=[0.5, 0.5])

    if current_state == MessagePerformative.PROPOSE:
        item_name = get_item_name(input)
        item = [x for x in agent.list_items if x.get_name() == item_name][0]

        if preferences.is_item_among_top_10_percent(item, agent.list_items):
            return MessagePerformative.ACCEPT

        return MessagePerformative.ASK_WHY

    if current_state == MessagePerformative.ACCEPT:
        return MessagePerformative.COMMIT

    if (
        current_state == MessagePerformative.ARGUE
        or current_state == MessagePerformative.BECAUSE
    ):
        item_name = get_item_name(input)
        proposed_argument: Argument = input.get_content()
        argument = agent.parse_argument(proposed_argument)
        if argument is None:
            if proposed_argument.decision:
                return MessagePerformative.ACCEPT
            else:
                return MessagePerformative.QUERY_REF

        return MessagePerformative.ARGUE
