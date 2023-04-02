
from mesa import Model

from agent.CommunicatingAgent import CommunicatingAgent
from conversational_model.FSM import FiniteStateMachine
from preferences.PreferenceModel import RandomIntervalProfile
from preferences.Preferences import Preferences
from preferences.CriterionName import CriterionName
from preferences.Item import Item
from preferences.CriterionValue import CriterionValue
from message.Message import Message
from message.MessagePerformative import MessagePerformative
from arguments.Argument import Argument

from typing import Callable, Dict, List, Union
import pandas as pd
import numpy as np


class ArgumentAgent(CommunicatingAgent):
    """ ArgumentAgent which inherit from CommunicatingAgent .
        The ArgumentAgent class is an agent that communicates with other agents and makes decisions based on their conversations. 
        It is a subclass of the CommunicatingAgent class from the mesa library.

        The class has the following attributes:
            preferences: a Preferences object that represents the agent's preference model.
            list_items: a list of items the agent can see.
            bag: a list of items that the agent possess.
            conversations: a dictionary that maps conversation IDs to FiniteStateMachine objects.
                           Each conversation is managed by a FSM, that controls the state of the protocol.
                           The FSM is initialized when the agent receives a message from another agent, or
                           when the agent initiates a conversation with another agent. Each agents maintains
                           a copy of the FSM for each conversation it is involved in. 

    """

    def __init__(self, unique_id: int, model: Model, name: str, decision_function: Callable[[Message, MessagePerformative], Message], message_builder: Callable[[Message, MessagePerformative], Message]):
        super().__init__(unique_id, model, name)
        self.preferences = Preferences(
            lambda preferences, input, current_state, next_states: decision_function(
                self, preferences, input, current_state, next_states),
            lambda preferences, input, next_states: message_builder(self, preferences, input, next_states))

        self.list_items: List[Item] = []
        self.bag: List[ArgumentAgent] = []
        self.conversations: Dict[str, FiniteStateMachine] = {}

    def step(self):
        super().step()
        print(f'Agent {self.get_name()} has: ', [str(x) for x in self.bag])
        nouveaux_messages = self.get_new_messages()
        for new_message in nouveaux_messages:
            exp = new_message.get_exp()
            if exp not in self.conversations:
                self.conversations[exp] = FiniteStateMachine(
                    self.get_name(), exp, verbose=False)

            self.conversations[exp].step(
                input=new_message)
            self.conversations[exp].step(
                input=new_message, preferences=self.preferences)

        self.init_conversation()

    def reset_conversation(self):
        finished_talking = [
            conversation for conversation in self.conversations.values() if conversation.has_finished()]

        for conversation in finished_talking:
            conversation.reset()

    def set_bag(self, bag: List[Item]):
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
            conversation for conversation, fsm in self.conversations.items() if not fsm.is_start()]

        finished_talking = [
            conversation for conversation, fsm in self.conversations.items() if fsm.has_finished()]

        continuous_talkings = list(
            set(continuous_talkings) - set(finished_talking))

        possible_choices = [agent for agent in self.model.schedule.agents if agent.get_name(
        ) not in continuous_talkings and agent.get_name() != self.get_name()]

        # print('possible_choices', [x.get_name()
        # for x in possible_choices], 'agent: ', self.get_name())
        #print('self.conversations', self.conversations)
        if len(possible_choices) == 0:
            return
        chosen_agent = np.random.choice(possible_choices, 1, replace=False)[0]

        if chosen_agent not in self.conversations:
            self.conversations[chosen_agent.get_name()] = FiniteStateMachine(
                self.get_name(), chosen_agent.get_name(), verbose=False)

        self.conversations[chosen_agent.get_name()].step(input=Message(
            self.get_name(), chosen_agent.get_name(), MessagePerformative.IDLE, None), preferences=self.preferences)

    def print_preference_table(self):
        criterion_list = self.preferences.get_criterion_value_list()
        criterion_names = self.preferences.get_criterion_name_list()

        pref_table = {
            criterion.name: {}
            for criterion in criterion_names
        }

        for criterionValue in criterion_list:
            criterion = criterionValue.get_criterion_name().name
            item = criterionValue.get_item().get_name()
            value = criterionValue.get_value().name

            pref_table[criterion][item] = value

        pref_table = pd.DataFrame(pref_table)
        print('Preference table of ', self.get_name(), ':')
        print('Bag: ', [x.get_name() for x in self.bag])
        print(pref_table)

    def generate_preferences(self, list_items: list[Item], map_item_criterion: dict[Item, dict[CriterionName, Union[int, float]]], verbose: int = 0):
        """
        The generate_preferences method generates the agent's preference model based on a list of items
        and a dictionary that maps each item to its associated criteria and the criteria's value.
        The method creates a RandomIntervalProfile object based on the given criteria and values, 
        and sets it as the agent's preference model. The preference model is then printed using
        the print_preference_table method.

        Args:
            list_items (list[Item]):  A list of items to generate preferences for.
            map_item_criterion (dict[Item, dict[CriterionName, Union[int, float]]]): A dictionary that maps 
                each item to its associated criteria and the criteria's value.
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
                value = profiler.get_value_from_data(
                    item, CriterionName[criterion])
                self.preferences.add_criterion_value(
                    CriterionValue(item, CriterionName[criterion], value))

        if verbose == 2:
            self.print_preference_table()

    def support_proposal(self, item):
        """
        Used when the agent receives " ASK_WHY " after having proposed an item
        : param item : str - name of the item which was proposed
        : return : string - the strongest supportive argument
        """
        best_argument = Argument(True, item)
        item = [x for x in self.list_items if x.get_name() == item][0]
        list_arguments = best_argument.list_supporting_proposal(
            item, self.preferences)
        # les arguments sont ordonnés dans la liste
        best_argument.add_premiss_couple_values(
            list_arguments[0].criterion_name, list_arguments[0].value)
        return best_argument

    def attack_proposal(self, item):
        """
        Used when the agent receives " ASK_WHY " after having proposed an item
        : param item : str - name of the item which was proposed
        : return : string - the strongest supportive argument
        """
        best_argument = Argument(True, item)
        item = [x for x in self.list_items if x.get_name() == item][0]
        list_arguments = best_argument.list_attacking_proposal(
            item, self.preferences)
        # les arguments sont ordonnés dans la liste
        best_argument.add_premiss_couple_values(
            list_arguments[0].criterion_name, list_arguments[0].value)
        return best_argument
