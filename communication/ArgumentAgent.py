from collections.abc import Callable
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from agent.CommunicatingAgent import CommunicatingAgent
from arguments.Argument import Argument
from arguments.CoupleValue import CoupleValue
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
from preferences.Value import Value


class Argumentation:
    def __init__(self, agent_a, agent_b):
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.G = nx.Graph()

    def add_argument(self, argument: Argument):
        self.G.add_node(argument)

        if argument.get_parent() is not None:
            self.G.add_edge(argument, argument.get_parent())

    def all_arguments(self):
        return self.G.nodes


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
        verbose: bool = False,
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
        self.agreed_items: Dict[str, List[str]] = {}
        self.proposed_items: Dict[str, List[str]] = {}
        self.conversations: dict[str, FiniteStateMachine] = {}
        self.argumentations: dict[str, Argumentation] = {}
        self.verbose = verbose

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
        self.agreed_items = bag

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
        df_items = pd.DataFrame(
            {x: [x.get_score(self.preferences)] for x in self.list_items},
        )
        print("Score of each item for ", self.get_name(), ":")
        print(df_items)

    def generate_preferences(
        self,
        list_items: list[Item],
        map_item_criterion: dict[Item, dict[CriterionName, int | float]],
        verbose: bool = False,
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
        if self.verbose:
            print("Agent ", self.get_name(), " criterion_name_list: ", end=" ")
            for criterion in criterion_name_list[0:-1]:
                print(criterion.name + " >", end=" ")
            print(criterion_name_list[-1].name)

        self.preferences.set_criterion_name_list(criterion_name_list)

        profiler = RandomIntervalProfile(map_item_criterion, verbose)
        # profiler = IntervalProfileCSV(map_item_criterion, verbose)

        for criterion in criterion_list:
            for item in list_items:
                value = profiler.get_value_from_data(item, CriterionName[criterion])
                self.preferences.add_criterion_value(
                    CriterionValue(item, CriterionName[criterion], value),
                )

        if verbose:
            self.print_preference_table()

    def support_proposal(self, item: str, agent: str):
        """
        Used when the agent receives " ASK_WHY " after having proposed an item
        : param item : str - name of the item which was proposed
        : return : string - the strongest supportive argument
        """
        items = [x for x in self.list_items if x.get_name() == item]
        item = items[0]
        best_argument = Argument(True, item, self.get_name())
        list_arguments = best_argument.list_supporting_proposal(item, self.preferences)
        # remove already used arguments
        if len(list_arguments) == 0:
            return None
        idx = 0
        best_argument.add_premiss_couple_values(
            list_arguments[idx].criterion_name,
            list_arguments[idx].value,
        )
        if agent in self.argumentations:
            already_used_arguments = self.argumentations[agent].all_arguments()
            while self.has_already_been_used(
                best_argument,
                already_used_arguments,
            ):
                idx += 1
                if idx >= len(list_arguments):
                    return None
                best_argument = Argument(True, item, self.get_name())
                best_argument.add_premiss_couple_values(
                    list_arguments[idx].criterion_name,
                    list_arguments[idx].value,
                )

        # les arguments sont ordonnés dans la liste
        return best_argument

    def has_already_been_used(
        self,
        best_argument: Argument,
        already_used_arguments: Argument,
    ) -> bool:
        for already_used_argument in already_used_arguments:
            if best_argument == already_used_argument:
                return True
        return False

    def attack_proposal(self, item):
        """
        Used when the agent receives " ASK_WHY " after having proposed an item
        : param item : str - name of the item which was proposed
        : return : string - the strongest supportive argument
        """
        item = [x for x in self.list_items if x.get_name() == item][0]
        best_argument = Argument(False, item, self.get_name())
        list_arguments = best_argument.list_attacking_proposal(item, self.preferences)
        # les arguments sont ordonnés dans la liste
        best_argument.add_premiss_couple_values(
            list_arguments[0].criterion_name,
            list_arguments[0].value,
        )
        return best_argument

    def better_alternative_same_criterion(
        self,
        proposed_item: Item,
        premisse: CoupleValue,
    ) -> Tuple[Item, Value]:
        proposed_value = premisse.value
        criterion_name = premisse.criterion_name
        for item in self.list_items:
            if item.get_name() == proposed_item.get_name():
                continue
            item_value = self.preferences.get_value(item, criterion_name)
            if item_value.value > proposed_value.value:
                return (item, item_value)
        return None

    def has_bad_evaluation(self, item: Item, premisse: CoupleValue):
        item_val = self.preferences.get_value(item, premisse.criterion_name)
        if item_val.value < premisse.value.value:
            return (item, item_val)
        return None

    def bad_evaluation_other_criterion(self, item: Item, premisse: CoupleValue):
        for criterion in self.preferences.get_criterion_name_list():
            if premisse.criterion_name == criterion:
                break
            item_value = self.preferences.get_value(item, criterion)
            if item_value.value < premisse.value.value:
                return (criterion, item_value)
        return None

    def parse_argument(self, argument: Argument):
        proposed_item = argument.get_item()
        comparison_premisses, couple_value_premisses = argument.get_premisses()
        if not argument.decision:
            best_argument = self.support_proposal(
                proposed_item.get_name(),
                argument.get_agent(),
            )
            return best_argument

        for premisse in couple_value_premisses:
            better_alternative = self.better_alternative_same_criterion(
                proposed_item,
                premisse,
            )
            bad_evaluation = self.has_bad_evaluation(proposed_item, premisse)
            bad_evaluation_other_criterion = self.bad_evaluation_other_criterion(
                proposed_item,
                premisse,
            )
            if better_alternative:
                reply = Argument(
                    True,
                    better_alternative[0],
                    self.get_name(),
                )
                reply.add_premiss_couple_values(
                    premisse.criterion_name,
                    better_alternative[1],
                )
                return reply
            if bad_evaluation:
                reply = Argument(not argument.decision, proposed_item, self.get_name())
                reply.add_premiss_couple_values(
                    premisse.criterion_name,
                    bad_evaluation[1],
                )
                return reply
            if bad_evaluation_other_criterion:
                reply = Argument(not argument.decision, proposed_item, self.get_name())
                reply.add_premiss_couple_values(
                    bad_evaluation_other_criterion[0],
                    bad_evaluation_other_criterion[1],
                )
                reply.add_premiss_comparison(
                    bad_evaluation_other_criterion[0],
                    premisse.criterion_name,
                )
                return reply
        return None
