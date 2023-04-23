from collections.abc import Callable
from typing import Dict, List

import numpy as np
import pandas as pd
from agent.CommunicatingAgent import CommunicatingAgent
from arguments.Argument import Argument
from conversational_model.FSM import FiniteStateMachine
from mesa import Model
from message.Message import Message
from message.MessagePerformative import MessagePerformative
from preferences.CriterionName import CriterionName
from preferences.CriterionValue import CriterionValue
from preferences.Item import Item
from preferences.PreferenceModel import (
    IntervalProfileCSV,  # noqa: F401
    RandomIntervalProfile,
)
from preferences.Preferences import Preferences


class ArgumentAgent(CommunicatingAgent):
    """ArgumentAgent which inherit from CommunicatingAgent .
    The ArgumentAgent class is an agent that communicates with other agents and makes
    decisions based on their conversations.
    It is a subclass of the CommunicatingAgent class from the mesa library.

    The class has the following attributes:
        preferences: a Preferences object that represents the agent's preference model.
        list_items: a list of items the agent can see.
        bag: a list of items that the agent possess.
        conversations:
        a dictionary that maps conversation IDs to FiniteStateMachine objects.
        Each conversation is managed by a FSM, that controls the state of the protocol.
        The FSM is initialized when the agent receives a message from another agent, or
        when the agent initiates a conversation with another agent.
        Each agents maintains a copy of the FSM for each conversation it is involved in.

    """

    def __init__(
        self,
        unique_id: int,
        model: Model,
        name: str,
        decision_function: Callable[[Message, MessagePerformative], Message],
        message_builder: Callable[[Message, MessagePerformative], Message],
    ):
        super().__init__(unique_id, model, name)
        self.preferences = Preferences(
            lambda preferences, input, current_state, next_states: decision_function(
                self,
                preferences,
                input,
                current_state,
                next_states,
            ),
            lambda preferences, input, next_states: message_builder(
                self,
                preferences,
                input,
                next_states,
            ),
        )

        self.list_items: list[Item] = []
        self.bag: Dict[str, List[str]] = {}
        self.conversations: dict[str, FiniteStateMachine] = {}

    def step(self):
        super().step()
        nouveaux_messages = self.get_new_messages()
        for new_message in nouveaux_messages:
            exp = new_message.get_exp()
            if exp not in self.conversations:
                self.conversations[exp] = FiniteStateMachine(
                    self.get_name(),
                    exp,
                    verbose=False,
                )

            # Infer the other agent action
            self.conversations[exp].step(input=new_message)

            # Do my action
            self.conversations[exp].step(
                input=new_message,
                preferences=self.preferences,
            )

        self.init_conversation()

    def reset_conversation(self):
        finished_talking = [
            conversation
            for conversation in self.conversations.values()
            if conversation.has_finished()
        ]

        for conversation in finished_talking:
            conversation.reset()

    def set_bag(self, bag: Dict[str, List[str]]):
        self.bag = bag

    def init_conversation(self):
        """Initialize a new conversation with another agent.

        This method selects a random agent that is not already engaged in a conversation
        with the current agent, and starts a new conversation with them using a new
        `FiniteStateMachine` object. If there are no available agents to converse with,
        this method does nothing.

        If the selected agent is already involved in a conversation with the current
        agent, no new conversation is started.

        This method is called at each step of the model, after processing incoming
        messages and resetting finished conversations.
        """

        continuous_talkings = [
            conversation
            for conversation, fsm in self.conversations.items()
            if not fsm.is_start()
        ]

        finished_talking = [
            conversation
            for conversation, fsm in self.conversations.items()
            if fsm.has_finished()
        ]

        continuous_talkings = list(set(continuous_talkings) - set(finished_talking))

        possible_choices = [
            agent
            for agent in self.model.schedule.agents
            if agent.get_name() not in continuous_talkings
            and agent.get_name() != self.get_name()
        ]

        # print('possible_choices', [x.get_name()
        # for x in possible_choices], 'agent: ', self.get_name())
        # print('self.conversations', self.conversations)
        if len(possible_choices) == 0:
            return
        chosen_agent = np.random.choice(possible_choices, 1, replace=False)[0]

        if chosen_agent not in self.conversations:
            self.conversations[chosen_agent.get_name()] = FiniteStateMachine(
                self.get_name(),
                chosen_agent.get_name(),
                verbose=False,
            )

        self.conversations[chosen_agent.get_name()].step(
            input=Message(
                self.get_name(),
                chosen_agent.get_name(),
                MessagePerformative.IDLE,
                None,
            ),
            preferences=self.preferences,
        )

    def print_preference_table(self):
        criterion_list = self.preferences.get_criterion_value_list()
        criterion_names = self.preferences.get_criterion_name_list()

        pref_table = {criterion.name: {} for criterion in criterion_names}

        for criterion_value in criterion_list:
            criterion = criterion_value.get_criterion_name().name
            item = criterion_value.get_item().get_name()
            value = criterion_value.get_value().name

            pref_table[criterion][item] = value

        pref_table = pd.DataFrame(pref_table)
        print("Preference table of ", self.get_name(), ":")
        print(pref_table)

    def generate_preferences(
        self,
        list_items: list[Item],
        map_item_criterion: dict[Item, dict[CriterionName, int | float]],
        verbose: int = 0,
    ):
        """
        The generate_preferences method generates the agent's preference
        model based on a list and a dictionary that maps each item to its
        associated criteria and the criteria's value. The method creates a
        RandomIntervalProfile object based on the given criteria and values,
        and sets it as the agent's preference model. The preference model is
        then printed using the print_preference_table method.

        Args:
            list_items (list[Item]):  A list of items to generate preferences for.
            map_item_criterion (dict[Item, dict[CriterionName, Union[int, float]]]):
            A dictionary that maps each item to its associated criteria.
        """

        self.list_items = list_items
        criterion_list = list(list(map_item_criterion.items())[0][1].keys())

        criterion_name_list = [CriterionName[x] for x in criterion_list]
        np.random.shuffle(criterion_name_list)
        self.preferences.set_criterion_name_list(criterion_name_list)

        profiler = RandomIntervalProfile(map_item_criterion, verbose)
        # profiler = IntervalProfileCSV(map_item_criterion, verbose)

        for criterion in criterion_list:
            for item in list_items:
                value = profiler.get_value_from_data(item, CriterionName[criterion])
                self.preferences.add_criterion_value(
                    CriterionValue(item, CriterionName[criterion], value),
                )

        if verbose == 2:
            self.print_preference_table()

    def support_proposal(self, item):
        """
        Used when the agent receives " ASK_WHY " after having proposed an item
        : param item : str - name of the item which was proposed
        : return : string - the strongest supportive argument
        """
        item = [x for x in self.list_items if x.get_name() == item][0]
        best_argument = Argument(True, item)
        list_arguments = best_argument.list_supporting_proposal(item, self.preferences)

        # les arguments sont ordonnés dans la liste
        best_argument.add_premiss_couple_values(
            list_arguments[0].criterion_name,
            list_arguments[0].value,
        )
        return best_argument

    def attack_proposal(self, item):
        """
        Used when the agent receives " ASK_WHY " after having proposed an item
        : param item : str - name of the item which was proposed
        : return : string - the strongest supportive argument
        """
        item = [x for x in self.list_items if x.get_name() == item][0]
        best_argument = Argument(False, item)
        list_arguments = best_argument.list_attacking_proposal(self.preferences)
        # les arguments sont ordonnés dans la liste
        best_argument.add_premiss_couple_values(
            list_arguments[0].criterion_name,
            list_arguments[0].value,
        )
        return best_argument

    def parse_argument(self, argument: Argument):
        proposed_item = argument.get_item()
        comparison_premisses, couple_value_premisses = argument.get_premisses()

        items_with_better_value = []
        criterions_preffered = []
        value_is_bad_for_criterion = []
        for premisse in couple_value_premisses:
            proposed_criterion = premisse.criterion_name
            proposed_value = premisse.value
            for criterion in self.preferences.get_criterion_name_list():
                # Check if one criterion is preffered
                if self.preferences.is_preferred_criterion(
                    criterion,
                    proposed_criterion,
                ):
                    criterions_preffered.append((criterion, proposed_criterion))

                    # Check if other criterion has a low value
                    items_value = self.preferences.get_value(proposed_item, criterion)
                    if items_value < proposed_value:
                        value_is_bad_for_criterion.append(
                            (criterion, items_value),
                        )

            # Check if one item has a better value for the argument's criterion
            for item in self.list_items:
                if item.get_name() == proposed_item.get_name():
                    continue

                if (
                    self.preferences.get_value(item, proposed_criterion)
                    > proposed_value
                ):
                    items_with_better_value.append((item, proposed_criterion))

        for criterion, value in value_is_bad_for_criterion:
            reply = Argument(False, proposed_item)
            reply.add_premiss_comparison(criterion, proposed_criterion)
            reply.add_premiss_couple_values(criterion, value)
            return reply

        # Check if my value is worse than the proposed value
        value = self.preferences.get_value(proposed_item, proposed_criterion)
        if value < proposed_value:
            reply = Argument(False, proposed_item)
            reply.add_premiss_couple_values(proposed_criterion, value)
            return reply

        for item, criterion in items_with_better_value:
            reply = Argument(True, item)
            reply.add_premiss_comparison(criterion, proposed_criterion)
            reply.add_premiss_couple_values(criterion, value)
            return reply
