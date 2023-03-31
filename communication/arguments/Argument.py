# !/ usr / bin / env python3

from arguments . Comparison import Comparison
from arguments . CoupleValue import CoupleValue
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

    def __init__(self, boolean_decision, item):
        """ Creates a new Argument .
        """
        self.decision = boolean_decision
        self.item = item
        self.comparison_list = []
        self.couple_value_list = []

    def add_premiss_comparison(self, criterion_name_1, criterion_name_2):
        """ Adds a premiss comparison in the comparison list .
        """
        self.comparison.append(Comparison(criterion_name_1, criterion_name_2))

    def add_premiss_couple_values(self, criterion_name, value):
        """ Add a premiss couple values in the couple values list .
        """
        self.couple_value_list.append(CoupleValue(criterion_name, value))

    def List_supporting_proposal(self, item, preferences):
        """ Generate a list of premisses which can be used to support an item
        : param item : Item - name of the item
        : return : list of all premisses PRO an item ( sorted by order of importance
        sed on agent ’s preferences )
        """
        resultat = []
        for criterion_name in preferences.get_criterion_name_list:
            value = preferences.get_value(self, item, criterion_name)
            if value == Value.VERY_GOOD or value == Value.GOOD:
                resultat.append(CoupleValue(criterion_name, value))
        return resultat

    def List_attacking_proposal(self, item, preferences):
        """ Generate a list of premisses which can be used to attack an item
        : param item : Item - name of the item
        : return : list of all premisses CON an item ( sorted by order of importance
        sed on preferences )
        """
        resultat = []
        for criterion_name in preferences.get_criterion_name_list:
            value = preferences.get_value(self, item, criterion_name)
            if value == Value.VERY_BAD or value == Value.BAD:
                resultat.append(CoupleValue(criterion_name, value))
        return resultat
