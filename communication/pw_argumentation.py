from mesa import Model
from mesa.time import RandomActivation

from agent.CommunicatingAgent import CommunicatingAgent
from preferences.PreferenceModel import RandomIntervalProfile, IntervalProfileCSV
from preferences.ItemFactory import ItemCreator_CSV
from preferences.Preferences import Preferences
from preferences.CriterionName import CriterionName
from preferences.Value import Value
from preferences.Item import Item
from preferences.CriterionValue import CriterionValue
from message.MessageService import MessageService
from message.Message import Message
from message.MessagePerformative import MessagePerformative
from arguments.Argument import Argument

from typing import List, Type, Union
import pandas as pd
import numpy as np
import random


class ArgumentAgent(CommunicatingAgent):
    """ ArgumentAgent which inherit from CommunicatingAgent .
    """

    def __init__(self, unique_id: int, model: Model, name: str, preferences: Preferences):
        super().__init__(unique_id, model, name)
        self.preferences = preferences
        self.list_items: List[Item] = []

    def step(self):
        super().step()
        if self.get_name() == "Agent 1":
            nouveaux_messages = self.get_new_messages()
            if nouveaux_messages == []:
                item = self.preferences.most_preferred(self.list_items)
                self.send_message(
                    Message(self.get_name(), "Agent 2", MessagePerformative.PROPOSE, item))
                print('Message de', self.get_name(),
                      'à Agent 2 : PROPOSE,', item)
            for message in nouveaux_messages:
                exp = message.get_exp()
                performative = message.get_performative()
                item = message.get_content()
                if performative == MessagePerformative.ACCEPT:
                    self.send_message(
                        Message(self.get_name(), exp, MessagePerformative.COMMIT, item))
                    print('Message de', self.get_name(),
                          'à', exp, ': COMMIT,', item)
                elif performative == MessagePerformative.COMMIT:
                    if item in self.list_items:
                        self.list_items.remove(item)
                        self.send_message(
                            Message(self.get_name(), exp, MessagePerformative.COMMIT, item))
                        print('Message de', self.get_name(),
                              'à', exp, ': COMMIT,', item)
                    else:
                        self.send_message(
                            Message(self.get_name(), exp, MessagePerformative.ARGUE, item))
                        print('Message de', self.get_name(),
                              'à', exp, ': ARGUE,', item)
                elif performative == MessagePerformative.ASK_WHY:
                    # A1 to A2: argue(item, premisses)
                    best_argument = self.support_proposal(item)
                    self.send_message(
                        Message(self.get_name(), exp, MessagePerformative.ARGUE, best_argument))
        if self.get_name() == "Agent 2":
            nouveaux_messages = self.get_new_messages()
            for message in nouveaux_messages:
                exp = message.get_exp()
                performative = message.get_performative()
                item = message.get_content()
                if performative == MessagePerformative.PROPOSE:
                    if self.preferences.is_item_among_top_10_percent(item, self.list_items):
                        self.send_message(
                            Message(self.get_name(), exp, MessagePerformative.ACCEPT, item))
                        print('Message de', self.get_name(),
                              'à', exp, ': ACCEPT,', item)
                    else:
                        self.send_message(
                            Message(self.get_name(), exp, MessagePerformative.ASK_WHY, item))
                        print('Message de', self.get_name(),
                              'à', exp, ': ASK_WHY,', item)
                elif performative == MessagePerformative.COMMIT:
                    if item in self.list_items:
                        self.list_items.remove(item)
                        self.send_message(
                            Message(self.get_name(), exp, MessagePerformative.COMMIT, item))
                        print('Message de', self.get_name(),
                              'à', exp, ': COMMIT,', item)
                    else:
                        self.send_message(
                            Message(self.get_name(), exp, MessagePerformative.ARGUE, item))
                        print('Message de', self.get_name(),
                              'à', exp, ': ARGUE,', item)

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
        print(pref_table)
        print('')

    def generate_preferences(self, list_items: list[Item], map_item_criterion: dict[Item, dict[CriterionName, Union[int, float]]], verbose: bool = True):

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

        if verbose:
            self.print_preference_table()

    def support_proposal(self, item):
        """
        Used when the agent receives " ASK_WHY " after having proposed an item
        : param item : str - name of the item which was proposed
        : return : string - the strongest supportive argument
        """
        list_arguments = Argument.List_supporting_proposal(
            item, self.preferences)
        best_argument = Argument(True, item)
        # les arguments sont ordonnés dans la liste
        best_argument.add_premiss_couple_values(list_arguments[0])
        return best_argument


class ArgumentModel(Model):
    """ ArgumentModel which inherit from Model .
    """

    def __init__(self, num_agents: int = 2):
        self.schedule = RandomActivation(self)
        self.__messages_service = MessageService(self.schedule)

        itemCreator = ItemCreator_CSV()
        items_list, map_item_criterion = itemCreator.create()

        self.current_id = 0
        for i in range(num_agents):
            new_agent = self.__create_agent()
            new_agent.generate_preferences(
                items_list, map_item_criterion, verbose=True)
            self.schedule.add(new_agent)

        self.running = True

    def __create_agent(self) -> ArgumentAgent:
        return ArgumentAgent(self.next_id(), self, "Agent " + str(self.current_id), Preferences())

    def step(self):
        self.__messages_service.dispatch_messages()
        self.schedule.step()

    def run_n_steps(self, n: int):
        for _ in range(n):
            self.step()


if __name__ == "__main__":
    model = ArgumentModel()

    model.run_n_steps(3)
