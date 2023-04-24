from enum import Enum


class CriterionName(Enum):
    """CriterionName enum class.
    Enumeration containing the possible CriterionName.
    """

    DIFFICULTY = 0
    PROFESSOR = 1
    EVALUATION = 2
    CV_BUILDING = 3
    FLEXIBLE = 4

    __POSITIVE_CRITERION = [1, 3, 4]

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
