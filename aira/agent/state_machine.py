"""
Deterministic state machine — controls all agent task transitions.
Every transition is explicit. Illegal transitions raise exceptions.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from aira.agent.state import TaskState, TERMINAL_STATES

logger = logging.getLogger(__name__)


class IllegalTransitionError(Exception):
    """Raised when an illegal state transition is attempted."""
    pass


# Whitelist of allowed transitions: from_state -> [to_states]
ALLOWED_TRANSITIONS: dict[TaskState, list[TaskState]] = {
    TaskState.INIT: [TaskState.PLANNING],
    TaskState.PLANNING: [
        TaskState.VALIDATING,
        TaskState.CANCELLED,
        TaskState.FAILED,
    ],
    TaskState.VALIDATING: [
        TaskState.EXECUTING,
        TaskState.PLANNING,  # plan_invalid → replan
    ],
    TaskState.EXECUTING: [
        TaskState.FILTERING,
        TaskState.CANCELLED,
        TaskState.FAILED,
    ],
    TaskState.FILTERING: [
        TaskState.UPDATING,
        TaskState.FAILED,  # security_violation
    ],
    TaskState.UPDATING: [
        TaskState.PLANNING,  # needs_replan
        TaskState.CONFIRMING_COMPLETION,  # all_goals_met
    ],
    TaskState.CONFIRMING_COMPLETION: [
        TaskState.COMPLETED,  # user_confirms
        TaskState.PLANNING,  # user_rejects
    ],
    # Terminal states have no outgoing transitions
    TaskState.COMPLETED: [],
    TaskState.FAILED: [],
    TaskState.CANCELLED: [],
}


class StateMachine:
    """
    Deterministic state machine for a single task.
    Persists state to JSON. Logs every transition.
    """

    def __init__(self, task_id: str, log_dir: Path):
        self.task_id = task_id
        self._state = TaskState.INIT
        self._history: list[dict] = []
        self._state_file = log_dir / f"task_{task_id}_state.json"
        self._state_file.parent.mkdir(parents=True, exist_ok=True)

        # Try to resume from persisted state
        if self._state_file.exists():
            self._load()
        else:
            self._record_transition(None, TaskState.INIT, "initialised")

    @property
    def current_state(self) -> TaskState:
        return self._state

    def transition(self, to_state: TaskState, reason: str = "") -> None:
        """
        Transition to a new state.

        Args:
            to_state: The target state.
            reason: Human-readable reason for the transition.

        Raises:
            IllegalTransitionError: If the transition is not allowed.
        """
        from_state = self._state

        # Check if transition is legal
        allowed = ALLOWED_TRANSITIONS.get(from_state, [])
        if to_state not in allowed:
            raise IllegalTransitionError(
                f"Illegal transition: {from_state.value} → {to_state.value}. "
                f"Allowed: {[s.value for s in allowed]}"
            )

        # Record and execute transition
        self._record_transition(from_state, to_state, reason)
        self._state = to_state
        self._persist()

        logger.info(
            f"[Task {self.task_id}] "
            f"{from_state.value} → {to_state.value} ({reason})"
        )

    def is_terminal(self) -> bool:
        """Check if current state is terminal (no further transitions)."""
        return self._state in TERMINAL_STATES

    def get_history(self) -> list[dict]:
        """Get the full transition history."""
        return self._history.copy()

    def _record_transition(
        self,
        from_state: Optional[TaskState],
        to_state: TaskState,
        reason: str,
    ) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "from": from_state.value if from_state else None,
            "to": to_state.value,
            "reason": reason,
        }
        self._history.append(entry)

    def _persist(self) -> None:
        data = {
            "task_id": self.task_id,
            "current_state": self._state.value,
            "history": self._history,
        }
        self._state_file.write_text(json.dumps(data, indent=2))

    def _load(self) -> None:
        data = json.loads(self._state_file.read_text())
        self._state = TaskState(data["current_state"])
        self._history = data.get("history", [])
