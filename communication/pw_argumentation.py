from mesa import Model
from mesa.time import RandomActivation
from agent.CommunicatingAgent import CommunicatingAgent
from preferences.PreferenceModel import RandomIntervalProfile
from preferences.ItemFactory import ItemCreator_CSV
from preferences.Preferences import Preferences
from preferences.CriterionName import CriterionName
from preferences.Value import Value
from preferences.Item import Item
from preferences.CriterionValue import CriterionValue
from message.MessageService import MessageService
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

    def step(self):
        super().step()

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

    def generate_preferences(self, list_items: list[Item], map_item_criterion: dict[Item, dict[CriterionName, Union[int, float]]]):

        criterion_list = list(list(map_item_criterion.items())[0][1].keys())

        criterion_name_list = [CriterionName[x] for x in criterion_list]
        np.random.shuffle(criterion_name_list)

        self.preferences.set_criterion_name_list(criterion_name_list)

        profiler = RandomIntervalProfile(map_item_criterion)

        for criterion in criterion_list:
            for item in list_items:
                value = profiler.get_value_from_data(
                    item, CriterionName[criterion])
                self.preferences.add_criterion_value(
                    CriterionValue(item, CriterionName[criterion], value))


class ArgumentModel(Model):
    """ ArgumentModel which inherit from Model .
    """

    def __init__(self):
        self.schedule = RandomActivation(self)
        self.__messages_service = MessageService(self.schedule)

        itemCreator = ItemCreator_CSV()
        items_list, map_item_criterion = itemCreator.create()

        self.current_id = 0
        a_pref = Preferences()
        a = ArgumentAgent(unique_id=self.next_id(), model=self,
                          name="Agent A", preferences=a_pref)
        a.generate_preferences(items_list, map_item_criterion)
        a.print_preference_table()
        self.schedule.add(a)

        b_pref = Preferences()
        b = ArgumentAgent(self.next_id(), self, "Agent B", b_pref)
        b.generate_preferences(items_list, map_item_criterion)
        self.schedule.add(b)

        self.running = True

    def step(self):
        self.__messages_service.dispatch_messages()
        self.schedule.step()


if __name__ == "__main__":
    model = ArgumentModel()

    model.step()
