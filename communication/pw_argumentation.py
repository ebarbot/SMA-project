import copy
from mesa import Model
from mesa.time import RandomActivation

from agent.CommunicatingAgent import CommunicatingAgent
from conversational_model.FSM import FiniteStateMachine, Turn
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

from typing import Callable, Dict, List, Type, Union
import pandas as pd
import numpy as np
import random

VERBOSE = 0


class ArgumentAgent(CommunicatingAgent):
    """ ArgumentAgent which inherit from CommunicatingAgent .
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
                    self.get_name(), exp, verbose=VERBOSE)

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
        continuous_talkings = [
            conversation for conversation, fsm in self.conversations.items() if not fsm.is_start()]

        finished_talking = [
            conversation for conversation, fsm in self.conversations.items() if fsm.has_finished()]

        continuous_talkings = list(
            set(continuous_talkings) - set(finished_talking))

        possible_choices = [agent for agent in self.model.schedule.agents if agent.get_name(
        ) not in continuous_talkings and agent.get_name() != self.get_name()]

        if VERBOSE >= 1:
            print('possible_choices', [x.get_name()
                                       for x in possible_choices], 'agent: ', self.get_name())
            print('self.conversations', self.conversations)
        if len(possible_choices) == 0:
            return
        chosen_agent = np.random.choice(possible_choices, 1, replace=False)[0]

        if chosen_agent not in self.conversations:
            self.conversations[chosen_agent.get_name()] = FiniteStateMachine(
                self.get_name(), chosen_agent.get_name(), verbose=VERBOSE)

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

        self.list_items = list_items
        criterion_list = list(list(map_item_criterion.items())[0][1].keys())

        criterion_name_list = [CriterionName[x] for x in criterion_list]
        np.random.shuffle(criterion_name_list)
        """_summary_
        """
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


def agent_decision_builder(agent: ArgumentAgent, preferences: Preferences, input: Message, current_state: MessagePerformative, next_states: List[MessagePerformative]) -> MessagePerformative:
    if current_state == MessagePerformative.IDLE:
        # return MessagePerformative.PROPOSE
        # return np.random.choice([MessagePerformative.PROPOSE, MessagePerformative.IDLE], p=[0.1, 0.9])
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


def agent_message_builder(agent: ArgumentAgent, preferences: Preferences, input: Message, next_state: MessagePerformative) -> Message:

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
            new_agent.set_bag(list(np.array(copy.deepcopy(items_list))[np.random.randint(
                0, len(items_list), size=np.random.randint(len(items_list), size=1))]))
            new_agent.generate_preferences(
                copy.deepcopy(items_list), copy.deepcopy(map_item_criterion), verbose=2)
            self.schedule.add(new_agent)

        self.running = True

    def __create_agent(self) -> ArgumentAgent:
        return ArgumentAgent(self.next_id(), self, "Agent " + str(self.current_id), agent_decision_builder, agent_message_builder)

    def step(self):
        self.__messages_service.dispatch_messages()
        self.schedule.step()

    def run_n_steps(self, n: int):
        for _ in range(n):
            print('-'*80)
            self.step()


if __name__ == "__main__":
    model = ArgumentModel(num_agents=5)

    model.run_n_steps(20)
