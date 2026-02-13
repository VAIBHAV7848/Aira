"""Tests for cost_controller security module."""

import pytest
from aira.security.cost_controller import CostController, BudgetExceededError


def test_estimate_cost():
    cc = CostController(max_cost_usd=1.00)
    cost = cc.estimate_cost("gpt-4o-mini", input_tokens=1000, output_tokens=500)
    assert cost > 0
    assert cost < 0.01  # gpt-4o-mini is cheap


def test_budget_exceeded():
    cc = CostController(max_cost_usd=0.01)
    cc.record_cost(0.009)
    with pytest.raises(BudgetExceededError):
        cc.check_budget(0.005)


def test_budget_ok():
    cc = CostController(max_cost_usd=1.00)
    cc.check_budget(0.001)  # Should not raise


def test_warning_at_threshold(caplog):
    import logging
    with caplog.at_level(logging.WARNING):
        cc = CostController(max_cost_usd=1.00, warning_threshold=0.80)
        cc.record_cost(0.79)
        cc.check_budget(0.02)  # 0.81 = 81% > 80% threshold
    assert "Cost warning" in caplog.text


def test_get_summary():
    cc = CostController(max_cost_usd=1.00)
    cc.record_cost(0.25)
    summary = cc.get_summary()
    assert summary["total_spent"] == 0.25
    assert summary["remaining"] == 0.75
    assert summary["call_count"] == 1
    assert summary["warning"] is False


def test_reset():
    cc = CostController(max_cost_usd=1.00)
    cc.record_cost(0.50)
    cc.reset()
    assert cc.total_cost == 0.0
    assert cc.call_count == 0


def test_persist_and_load(tmp_path):
    persist_file = tmp_path / "cost.json"
    cc1 = CostController(max_cost_usd=1.00, persist_path=persist_file)
    cc1.record_cost(0.33)

    cc2 = CostController(max_cost_usd=1.00, persist_path=persist_file)
    assert cc2.total_cost == 0.33
    assert cc2.call_count == 1
