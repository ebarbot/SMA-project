from mesa import Model
from mesa.time import RandomActivation
from agent.CommunicatingAgent import CommunicatingAgent
from message.MessageService import MessageService
from preferences import Preferences, Item, CriterionName, CriterionValue, Value
from typing import List
import random
import inspect

class GenerateData(object):
    def __init__(self, num_items: int):
        self.num_items = num_items
        self.list_items = None
        self.criteria_names = None

    def generate_items(self):
        self.list_items = []
        self.items_values = []
        for i in range(self.num_items):
            new_item = Item(f"Item {i}", "This is an item :)")
            self.list_items.append(new_item)
        

        return self.list_items
    
    def generate_criteria(self):
        self.criteria_list = []

        # https://stackoverflow.com/questions/9058305/getting-attributes-of-a-class
        attributes = inspect.getmembers(CriterionName, lambda a: not(inspect.isroutine(a)))
        self.criteria_names = [a for a in attributes if not(a[0].startswith('__') and a[0].endswith('__'))]

        return self.criteria_names


class ArgumentAgent (CommunicatingAgent):
    """ ArgumentAgent which inherit from CommunicatingAgent .
    """
    def __init__ (self, unique_id: int, model: Model, name: str, preferences: Preferences):
        super ().__init__ (unique_id, model, name, preferences)
        self.preference = preferences

    def step (self):
        super ().step ()

    def get_preference (self):
        return self.preference

    def generate_preferences(self, list_items, criterion_list):
        self.preferences.set_criterion_name_list(random.shuffle([x[0] for x in criterion_list]))
        for criterion in criterion_list:
            criterion_name,max_value = criterion
            p1p2p3 = [max_value*random.random() for i in range(3)]
            p1p2p3.sort()
            for item in list_items:
                real_value = item[criterion_name]
                if real_value<p1p2p3[0]:
                    value=Value.VERY_BAD
                elif real_value<p1p2p3[1]:
                    value=Value.BAD
                elif real_value<p1p2p3[2]:
                    value=Value.GOOD
                else:
                    value=Value.VERY_GOOD
                self.add_criterion_value(CriterionValue(item, criterion_name, value))

class ArgumentModel (Model):
    """ ArgumentModel which inherit from Model .
    """
    def __init__(self):
        self.schedule = RandomActivation(self)
        self.__messages_service = MessageService(self.schedule)

        # ICED 12330  6.3  3.8  4.8  65
        # E    17100  0    3    2.2  48

        a = ArgumentAgent (id, "Agent A")
        a.generate_preferences()
        self.schedule.add(a)
        b = ArgumentAgent (id, "Agent B")
        b.generate_preferences()
        self.schedule.add(b)

        self.running = True

    def step (self):
        self.__messages_service.dispatch_messages ()
        self.schedule.step ()


if __name__ == " __main__ ":
    argument_model = ArgumentModel ()

    # To be completed