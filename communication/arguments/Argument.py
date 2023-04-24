from typing import List, Tuple

from arguments.Comparison import Comparison
from arguments.CoupleValue import CoupleValue
from preferences.Item import Item
from preferences.Preferences import Preferences
from preferences.Value import Value


class Argument:
    """Argument class .
    This class implements an argument used during the interaction .

    attr :
        decision :
        item :
        comparison_list :
        couple_values_list :
    """

    def __init__(self, boolean_decision: bool, item: Item, agent_name: str):
        """Creates a new Argument ."""
        self.decision = boolean_decision
        self.__item: Item = item
        self.__comparison_list: List[Comparison] = []
        self.__couple_value_list: List[CoupleValue] = []
        self.__parent: "Argument" = None
        self.__agent_name: str = agent_name

    def __hash__(self) -> int:
        return hash(self.__str__())

    def __eq__(self, __value: object) -> bool:
        return self.__str__() == __value.__str__()

    def get_agent(self) -> str:
        return self.__agent_name

    def set_parent(self, parent: "Argument"):
        self.__parent = parent

    def get_parent(self) -> "Argument":
        return self.__parent

    def get_premiss_comparison(self) -> List[Comparison]:
        """Returns the comparison list ."""
        return self.__comparison_list

    def get_premiss_couple_values(self) -> List[CoupleValue]:
        """Returns the couple values list ."""
        return self.__couple_value_list

    def get_item(self):
        return self.__item

    def get_premisses(self) -> Tuple[List[Comparison], List[CoupleValue]]:
        """Returns the couple values list ."""
        return self.__comparison_list, self.__couple_value_list

    def __str__(self) -> str:
        """Returns a string representation of the argument ."""
        return (
            f"[ {'not ' if not self.decision else ''}"
            + str(self.__item)
            + "; "
            + str(",".join([str(x) for x in self.__couple_value_list]))
            + " "
            + f"{'and ' + str(' '.join([str(x) for x in self.__comparison_list])) if len(self.__comparison_list) > 0 else ''}"
            + "] "
        )

    def add_premiss_comparison(self, criterion_name_1, criterion_name_2):
        """Adds a premiss comparison in the comparison list ."""
        self.__comparison_list.append(Comparison(criterion_name_1, criterion_name_2))

    def add_premiss_couple_values(self, criterion_name, value):
        """Add a premiss couple values in the couple values list ."""
        self.__couple_value_list.append(CoupleValue(criterion_name, value))

    def list_proposals(self, preferences: Preferences):
        """Generate a list of premisses which can be used to support or attack an item
        : return : list of all premisses PRO or CON"""
        if self.decision:
            self.list_supporting_proposal(preferences)

        self.list_attacking_proposal(preferences)

    def list_supporting_proposal(
        self,
        item: Item,
        preferences: Preferences,
    ) -> List[CoupleValue]:
        """Generate a list of premisses which can be used to support an item
        : param item : Item - name of the item
        : return : list of all premisses PRO an item ( sorted by order of importance
        based on agent's preferences )
        """
        supporting_proposals = []
        criterion_names = preferences.get_criterion_name_list()
        for criterion_name in criterion_names:
            value = preferences.get_value(item, criterion_name)

            if value == Value.VERY_GOOD or value == Value.GOOD:
                supporting_proposals.append(CoupleValue(criterion_name, value))

        return supporting_proposals

    def list_attacking_proposal(
        self,
        item: Item,
        preferences: Preferences,
    ) -> List[CoupleValue]:
        """Generate a list of premisses which can be used to attack an item
        : return : list of all premisses CON an item ( sorted by order of importance
        based on preferences )
        """
        attacking_proposals = []
        criterion_names = preferences.get_criterion_name_list()
        for criterion_name in criterion_names:
            value = preferences.get_value(item, criterion_name)

            if value == Value.VERY_BAD or value == Value.BAD:
                attacking_proposals.append(CoupleValue(criterion_name, value))

        return attacking_proposals
