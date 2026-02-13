# Phase 3 — State Machine

## Goal
Build the deterministic FSM that controls the entire agent loop.

## Prerequisites
- Phase 2 completed (all security tests passing)

## Files to Create

### `aira/agent/state.py`

```python
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
```

---

### `aira/agent/state_machine.py`

```python
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
```

---

## Test File

### `aira/tests/test_state_machine.py`

```python
"""Tests for the deterministic state machine."""

import pytest
from aira.agent.state import TaskState, TERMINAL_STATES
from aira.agent.state_machine import StateMachine, IllegalTransitionError


@pytest.fixture
def sm(tmp_path):
    return StateMachine("test-001", tmp_path)


def test_initial_state(sm):
    assert sm.current_state == TaskState.INIT


def test_valid_transition_init_to_planning(sm):
    sm.transition(TaskState.PLANNING, "start")
    assert sm.current_state == TaskState.PLANNING


def test_full_happy_path(sm):
    sm.transition(TaskState.PLANNING, "start")
    sm.transition(TaskState.VALIDATING, "plan ready")
    sm.transition(TaskState.EXECUTING, "plan valid")
    sm.transition(TaskState.FILTERING, "tool done")
    sm.transition(TaskState.UPDATING, "output safe")
    sm.transition(TaskState.CONFIRMING_COMPLETION, "all goals met")
    sm.transition(TaskState.COMPLETED, "user confirms")
    assert sm.current_state == TaskState.COMPLETED
    assert sm.is_terminal()


def test_illegal_transition_raises(sm):
    with pytest.raises(IllegalTransitionError):
        sm.transition(TaskState.EXECUTING)  # can't go INIT→EXECUTING


def test_no_transition_from_terminal(sm):
    sm.transition(TaskState.PLANNING)
    sm.transition(TaskState.CANCELLED, "user cancel")
    with pytest.raises(IllegalTransitionError):
        sm.transition(TaskState.PLANNING)


def test_replan_cycle(sm):
    sm.transition(TaskState.PLANNING, "start")
    sm.transition(TaskState.VALIDATING, "plan ready")
    sm.transition(TaskState.PLANNING, "plan invalid - replan")
    assert sm.current_state == TaskState.PLANNING


def test_history_recorded(sm):
    sm.transition(TaskState.PLANNING, "start")
    history = sm.get_history()
    assert len(history) >= 2  # INIT + PLANNING
    assert history[-1]["to"] == "PLANNING"


def test_persist_and_reload(tmp_path):
    sm1 = StateMachine("persist-test", tmp_path)
    sm1.transition(TaskState.PLANNING, "start")
    sm1.transition(TaskState.VALIDATING, "ready")

    sm2 = StateMachine("persist-test", tmp_path)
    assert sm2.current_state == TaskState.VALIDATING


def test_terminal_states():
    assert TaskState.COMPLETED in TERMINAL_STATES
    assert TaskState.FAILED in TERMINAL_STATES
    assert TaskState.CANCELLED in TERMINAL_STATES
    assert TaskState.PLANNING not in TERMINAL_STATES
```

## Verification

```powershell
pytest aira/tests/test_state_machine.py -v
```

## Done When
- [ ] `aira/agent/state.py` created
- [ ] `aira/agent/state_machine.py` created
- [ ] `aira/tests/test_state_machine.py` created
- [ ] All tests pass

## Next Phase
→ `phase_04_tools.md`
