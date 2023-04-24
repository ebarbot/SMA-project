from enum import Enum


class CriterionName(Enum):
    """CriterionName enum class.
    Enumeration containing the possible CriterionName.
    """

    PRODUCTION_COST = 0
    CONSUMPTION = 1
    DURABILITY = 2
    ENVIRONMENT_IMPACT = 3
    NOISE = 4

    __POSITIVE_CRITERION = [2]

    @staticmethod
    def is_positive_criterion(criterion: "CriterionName") -> bool:
        """Get the direction of the criterion.
        Returns:
            bool: True if the criterion is a benefit, False otherwise.
        """
        if criterion.value in CriterionName.__POSITIVE_CRITERION:
            return True
        else:
            return False
