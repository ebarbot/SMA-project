# !/ usr / bin / env python3

from typing import List
from arguments.Comparison import Comparison
from arguments.CoupleValue import CoupleValue
from preferences.Item import Item
from preferences.Preferences import Preferences
from preferences.Value import Value


class Argument:
    """ Argument class .
    This class implements an argument used during the interaction .

    attr :
        decision :
        item :
        comparison_list :
        couple_values_list :
    """

    def __init__(self, boolean_decision, item: str):
        """ Creates a new Argument .
        """
        self.decision = boolean_decision
        self.item: str = item
        self.comparison_list = []
        self.couple_value_list = []

    def __str__(self) -> str:
        """ Returns a string representation of the argument .
        """
        return "Argument : " + str(self.item) + " " + str([str(x) for x in self.comparison_list]) + " " + str([str(x) for x in self.couple_value_list])

    def add_premiss_comparison(self, criterion_name_1, criterion_name_2):
        """ Adds a premiss comparison in the comparison list .
        """
        self.comparison_list.append(Comparison(
            criterion_name_1, criterion_name_2))

    def add_premiss_couple_values(self, criterion_name, value):
        """ Add a premiss couple values in the couple values list .
        """
        self.couple_value_list.append(CoupleValue(criterion_name, value))

    def list_supporting_proposal(self, item, preferences: Preferences) -> List[CoupleValue]:
        """ Generate a list of premisses which can be used to support an item
        : param item : Item - name of the item
        : return : list of all premisses PRO an item ( sorted by order of importance
        sed on agent ’s preferences )
        """
        resultat = []
        for criterion_name in preferences.get_criterion_name_list():
            value = preferences.get_value(item, criterion_name)
            if value == Value.VERY_GOOD or value == Value.GOOD:
                resultat.append(CoupleValue(criterion_name, value))
        return resultat

    def list_attacking_proposal(self, item, preferences: Preferences):
        """ Generate a list of premisses which can be used to attack an item
        : param item : Item - name of the item
        : return : list of all premisses CON an item ( sorted by order of importance
        sed on preferences )
        """
        resultat = []
        for criterion_name in preferences.get_criterion_name_list():
            value = preferences.get_value(item, criterion_name)
            if value == Value.VERY_BAD or value == Value.BAD:
                resultat.append(CoupleValue(criterion_name, value))
        return resultat
