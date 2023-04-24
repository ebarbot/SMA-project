#!/usr/bin/env python3

from enum import Enum


class MessagePerformative(Enum):
    """MessagePerformative enum class.
    Enumeration containing the possible message performative.
    """

    IDLE = 0
    FINISHED = 1
    ACK = 2
    REJECT = 3
    PROPOSE = 101
    ACCEPT = 102
    COMMIT = 103
    ASK_WHY = 104
    BECAUSE = 105
    ARGUE = 106
    QUERY_REF = 107
    INFORM_REF = 108

    def __str__(self):
        """Returns the name of the enum item."""
        return f"{self.name}"
