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
