# Phase 7 — External Planner

## Goal
Cost-controlled external LLM calls via litellm for complex planning tasks.

## Prerequisites
- Phase 2 security (cost_controller, outbound_guard) passing
- Phase 1 config passing

## File to Create

### `aira/llm/external_planner.py`

```python
"""
External planner — cost-controlled LLM calls via litellm.
Used only when local model needs help with complex multi-step reasoning.
"""

import json
import logging
from typing import Optional

import tiktoken
from litellm import acompletion

from aira.security.cost_controller import CostController, BudgetExceededError
from aira.security.outbound_guard import sanitise_for_llm

logger = logging.getLogger(__name__)

# Confidence threshold — below this, escalate to external LLM
CONFIDENCE_THRESHOLD = 0.6


class PlannerError(Exception):
    """Raised when external planning fails."""
    pass


class ExternalPlanner:
    """External LLM planner with cost control and sanitisation."""

    def __init__(
        self,
        cost_controller: CostController,
        model: str = "gpt-4o-mini",
        api_key: str = "",
        max_output_chars: int = 20000,
    ):
        self.cost_controller = cost_controller
        self.model = model
        self.api_key = api_key
        self.max_output_chars = max_output_chars

    async def plan(self, context: str, task_description: str) -> dict:
        """
        Generate a structured plan for a task.

        Args:
            context: Sanitised context (file contents, history, etc.)
            task_description: What the user wants done.

        Returns:
            dict with keys: steps (list), confidence (float), done (bool)

        Raises:
            BudgetExceededError: If cost cap would be exceeded.
            PlannerError: If the API call or parsing fails.
        """
        # 1. Sanitise context
        safe_context = sanitise_for_llm(context, self.max_output_chars)

        # 2. Estimate tokens
        try:
            encoder = tiktoken.encoding_for_model("gpt-4o")
        except Exception:
            encoder = tiktoken.get_encoding("cl100k_base")

        system_prompt = (
            "You are a task planner. Given a context and task, produce a JSON plan.\n"
            "Respond ONLY with valid JSON:\n"
            '{"steps": [{"tool": "tool_name", "params": {...}, "description": "..."}], '
            '"confidence": 0.0-1.0, "done": false}\n'
            "Available tools: file_read, file_write, python_run"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{safe_context}\n\nTask: {task_description}"},
        ]

        input_text = system_prompt + safe_context + task_description
        input_tokens = len(encoder.encode(input_text))
        estimated_output = 500
        estimated_cost = self.cost_controller.estimate_cost(
            self.model, input_tokens, estimated_output
        )

        # 3. Check budget
        self.cost_controller.check_budget(estimated_cost)

        # 4. Call external LLM
        try:
            response = await acompletion(
                model=self.model,
                messages=messages,
                api_key=self.api_key if self.api_key else None,
                temperature=0.2,
                max_tokens=1000,
            )

            content = response.choices[0].message.content
            actual_tokens = response.usage.total_tokens if response.usage else input_tokens + estimated_output
            actual_cost = self.cost_controller.estimate_cost(
                self.model, 
                response.usage.prompt_tokens if response.usage else input_tokens,
                response.usage.completion_tokens if response.usage else estimated_output,
            )

            # 5. Record cost
            self.cost_controller.record_cost(actual_cost)

            # 6. Parse response
            try:
                plan = json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code blocks
                if "```json" in content:
                    json_str = content.split("```json")[1].split("```")[0].strip()
                    plan = json.loads(json_str)
                elif "```" in content:
                    json_str = content.split("```")[1].split("```")[0].strip()
                    plan = json.loads(json_str)
                else:
                    raise PlannerError(f"Could not parse plan from response: {content[:200]}")

            # Validate plan structure
            if "steps" not in plan:
                plan["steps"] = []
            if "confidence" not in plan:
                plan["confidence"] = 0.5
            if "done" not in plan:
                plan["done"] = False

            logger.info(
                f"Plan generated: {len(plan['steps'])} steps, "
                f"confidence={plan['confidence']}, cost=${actual_cost:.6f}"
            )
            return plan

        except BudgetExceededError:
            raise  # Let it propagate
        except PlannerError:
            raise
        except Exception as e:
            raise PlannerError(f"External planner failed: {e}")

    def should_escalate(self, confidence: float, complexity: str = "low") -> bool:
        """
        Decide whether to escalate to the external planner.

        Args:
            confidence: Local model's confidence (0.0 - 1.0).
            complexity: Task complexity ("low", "medium", "high").

        Returns:
            True if external planner should be used.
        """
        if complexity == "high":
            return True
        if confidence < CONFIDENCE_THRESHOLD:
            return True
        return False
```

---

## Test File

### `aira/tests/test_external_planner.py`

```python
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
```

## Verification

```powershell
pytest aira/tests/test_external_planner.py -v
```

## Done When
- [ ] `aira/llm/external_planner.py` created
- [ ] `aira/tests/test_external_planner.py` created
- [ ] All tests pass (mocked)

## Next Phase
→ `phase_08_agent_loop.md`
