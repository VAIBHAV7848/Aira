"""Tests for external planner (mocked — no real API calls)."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from aira.llm.external_planner import ExternalPlanner, PlannerError
from aira.security.cost_controller import CostController, BudgetExceededError


@pytest.fixture
def cost_ctrl():
    return CostController(max_cost_usd=1.00)


def test_should_escalate_high_complexity(cost_ctrl):
    planner = ExternalPlanner(cost_ctrl)
    assert planner.should_escalate(confidence=0.9, complexity="high") is True


def test_should_escalate_low_confidence(cost_ctrl):
    planner = ExternalPlanner(cost_ctrl)
    assert planner.should_escalate(confidence=0.3, complexity="low") is True


def test_should_not_escalate(cost_ctrl):
    planner = ExternalPlanner(cost_ctrl)
    assert planner.should_escalate(confidence=0.8, complexity="low") is False


def test_budget_exceeded_before_call():
    cc = CostController(max_cost_usd=0.001)
    cc.record_cost(0.001)  # Already at limit
    planner = ExternalPlanner(cc)
    with pytest.raises(BudgetExceededError):
        import asyncio
        asyncio.run(planner.plan("context", "do something"))


@pytest.mark.asyncio
async def test_plan_returns_structured_output(cost_ctrl):
    """Mocked plan call should return structured dict."""
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(
            content='{"steps": [{"tool": "file_read", "params": {"path": "x.txt"}, "description": "read file"}], "confidence": 0.8, "done": false}'
        ))
    ]
    mock_response.usage = MagicMock(
        prompt_tokens=100, completion_tokens=50, total_tokens=150
    )

    with patch("aira.llm.external_planner.acompletion", new_callable=AsyncMock) as mock_api:
        mock_api.return_value = mock_response
        planner = ExternalPlanner(cost_ctrl)
        plan = await planner.plan("some context", "read a file")

    assert "steps" in plan
    assert len(plan["steps"]) == 1
    assert plan["steps"][0]["tool"] == "file_read"
    assert plan["confidence"] == 0.8
