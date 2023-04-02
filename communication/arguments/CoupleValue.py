# !/ usr / bin / env python3


from preferences.CriterionName import CriterionName
from preferences.CriterionValue import CriterionValue


class CoupleValue:
    """ CoupleValue class .
    This class implements a couple value used in argument object .

    attr :
        criterion_name :
        value :
    """

    def __init__(self, criterion_name: CriterionName, value: CriterionValue) -> None:
        """ Creates a new couple value .
        """
        self.criterion_name = criterion_name
        self.value = value

    def __str__(self) -> str:
        return f'({self.criterion_name.name}, {self.value.name})'
