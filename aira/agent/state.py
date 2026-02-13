"""Task states for the Aira agent."""

from enum import Enum


class TaskState(Enum):
    INIT = "INIT"
    PLANNING = "PLANNING"
    VALIDATING = "VALIDATING"
    EXECUTING = "EXECUTING"
    FILTERING = "FILTERING"
    UPDATING = "UPDATING"
    CONFIRMING_COMPLETION = "CONFIRMING_COMPLETION"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


# Terminal states — no transitions allowed out of these
TERMINAL_STATES = {TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED}
