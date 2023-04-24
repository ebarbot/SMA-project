import os
import sys
from math import ceil
from typing import Callable, List

import numpy as np
from preferences.CriterionName import CriterionName
from preferences.CriterionValue import CriterionValue
from preferences.Item import Item
from preferences.Value import Value

sys.path.append(os.getcwd())
from message.Message import Message  # nopep8  # noqa: E402
from message.MessagePerformative import MessagePerformative  # nopep8  # noqa: E402


class Preferences:
    """Preferences class.
    This class implements the preferences of an agent.

    attr:
        criterion_name_list: the list of criterion name (ordered by importance)
        criterion_value_list: the list of criterion value
    """

    def __init__(
        self,
        decision_function: Callable[[Message, MessagePerformative], Message],
        message_builder: Callable[[Message, MessagePerformative], Message],
    ):
        """Creates a new Preferences object."""
        self.__criterion_name_list: list[CriterionName] = []
        self.__criterion_value_list: list[CriterionValue] = []
        self.__decide: Callable[
            [Preferences, Message, MessagePerformative, List[MessagePerformative]],
            MessagePerformative,
        ] = decision_function

        self.__build_message: Callable[
            [Preferences, Message, MessagePerformative],
            MessagePerformative,
        ] = message_builder

    def decide(
        self,
        input: Message,
        current_state: MessagePerformative,
        next_states: List[MessagePerformative],
    ) -> MessagePerformative:
        return self.__decide(self, input, current_state, next_states)

    def build_message(self, input: Message, next_state: MessagePerformative) -> Message:
        return self.__build_message(self, input, next_state)

    def get_criterion_name_list(self) -> list[CriterionName]:
        """Returns the list of criterion name."""
        return self.__criterion_name_list

    def get_criterion_value_list(self) -> list[CriterionValue]:
        """Returns the list of criterion value."""
        return self.__criterion_value_list

    def set_criterion_name_list(self, criterion_name_list: list[CriterionName]) -> None:
        """Sets the list of criterion name."""
        self.__criterion_name_list = criterion_name_list

    def add_criterion_value(self, criterion_value: list[CriterionValue]) -> None:
        """Adds a criterion value in the list."""
        self.__criterion_value_list.append(criterion_value)

    def get_value(self, item: Item, criterion_name: CriterionName) -> Value:
        """Gets the value for a given item and a given criterion name."""
        for value in self.__criterion_value_list:
            if (
                value.get_item().get_name() == item.get_name()
                and value.get_criterion_name() == criterion_name
            ):
                return value.get_value()
        return None

    def is_preferred_criterion(
        self,
        criterion_name_1: CriterionName,
        criterion_name_2: CriterionName,
    ) -> bool:
        """Returns if a criterion 1 is preferred to the criterion 2."""
        for criterion_name in self.__criterion_name_list:
            if criterion_name == criterion_name_1:
                return True
            if criterion_name == criterion_name_2:
                return False

    def is_preferred_item(self, item_1: Item, item_2: Item) -> bool:
        """Returns if the item 1 is preferred to the item 2."""
        return item_1.get_score(self) > item_2.get_score(self)

    def most_preferred(self, item_list: list[Item]) -> Item:
        """Returns the most preferred item from a list."""
        liste = [(item.get_score(self), item) for item in item_list]
        max_val = liste[0][0]
        for val in liste:
            max_val = max(max_val, val[0])
        return np.random.choice([item[1] for item in liste if item[0] == max_val])

    # Sort the items by their score
    def sort_items(self, item_list: list[Item]) -> list[tuple[float, Item]]:
        return sorted(
            [(item.get_score(self), item.get_name()) for item in item_list],
            reverse=True,
        )

    def is_item_among_top_10_percent(self, item: Item, item_list: list[Item]) -> bool:
        """
        Return whether a given item is among the top 10 percent of the preferred items.

        :return: a boolean, True means that the item is among the favourite ones
        """
        sorted_list = self.sort_items(item_list)
        max_index = ceil(len(item_list) / 10) - 1
        return item.get_score(self) >= sorted_list[max_index][0]


if __name__ == "__main__":
    """Testing the Preferences class."""
    agent_pref = Preferences()
    agent_pref.set_criterion_name_list(
        [
            CriterionName.PRODUCTION_COST,
            CriterionName.ENVIRONMENT_IMPACT,
            CriterionName.CONSUMPTION,
            CriterionName.DURABILITY,
            CriterionName.NOISE,
        ]
    )

    diesel_engine = Item("Diesel Engine", "A super cool diesel engine")
    agent_pref.add_criterion_value(
        CriterionValue(diesel_engine, CriterionName.PRODUCTION_COST, Value.VERY_GOOD)
    )
    agent_pref.add_criterion_value(
        CriterionValue(diesel_engine, CriterionName.CONSUMPTION, Value.GOOD)
    )
    agent_pref.add_criterion_value(
        CriterionValue(diesel_engine, CriterionName.DURABILITY, Value.VERY_GOOD)
    )
    agent_pref.add_criterion_value(
        CriterionValue(diesel_engine, CriterionName.ENVIRONMENT_IMPACT, Value.VERY_BAD)
    )
    agent_pref.add_criterion_value(
        CriterionValue(diesel_engine, CriterionName.NOISE, Value.VERY_BAD)
    )

    electric_engine = Item("Electric Engine", "A very quiet engine")
    agent_pref.add_criterion_value(
        CriterionValue(electric_engine, CriterionName.PRODUCTION_COST, Value.BAD)
    )
    agent_pref.add_criterion_value(
        CriterionValue(electric_engine, CriterionName.CONSUMPTION, Value.VERY_BAD)
    )
    agent_pref.add_criterion_value(
        CriterionValue(electric_engine, CriterionName.DURABILITY, Value.GOOD)
    )
    agent_pref.add_criterion_value(
        CriterionValue(
            electric_engine, CriterionName.ENVIRONMENT_IMPACT, Value.VERY_GOOD
        )
    )
    agent_pref.add_criterion_value(
        CriterionValue(electric_engine, CriterionName.NOISE, Value.VERY_GOOD)
    )

    """test list of preferences"""
    print(diesel_engine)
    print(electric_engine)
    print(diesel_engine.get_value(agent_pref, CriterionName.PRODUCTION_COST))
    print(
        agent_pref.is_preferred_criterion(
            CriterionName.CONSUMPTION, CriterionName.NOISE
        )
    )
    print(
        "Electric Engine > Diesel Engine : {}".format(
            agent_pref.is_preferred_item(electric_engine, diesel_engine)
        )
    )
    print(
        "Diesel Engine > Electric Engine : {}".format(
            agent_pref.is_preferred_item(diesel_engine, electric_engine)
        )
    )
    print(
        "Electric Engine (for agent 1) = {}".format(
            electric_engine.get_score(agent_pref)
        )
    )
    print(
        "Diesel Engine (for agent 1) = {}".format(diesel_engine.get_score(agent_pref))
    )
    print(
        "Most preferred item is : {}".format(
            agent_pref.most_preferred([diesel_engine, electric_engine]).get_name()
        )
    )
