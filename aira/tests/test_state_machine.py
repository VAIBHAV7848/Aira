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
