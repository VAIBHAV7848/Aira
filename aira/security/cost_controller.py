"""
Cost controller — tracks and enforces spending limits for external API calls.
"""

import json
import logging
import os
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Approximate cost per 1K tokens (USD) — update as prices change
MODEL_COSTS: dict[str, dict[str, float]] = {
    "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "default": {"input": 0.01, "output": 0.03},
}


class BudgetExceededError(Exception):
    """Raised when the cost cap would be exceeded."""
    pass


class CostController:
    """Tracks and enforces per-task spending limits."""

    def __init__(
        self,
        max_cost_usd: float = 1.00,
        warning_threshold: float = 0.80,
        persist_path: Optional[Path] = None,
    ):
        self.max_cost_usd = max_cost_usd
        self.warning_threshold = warning_threshold
        self.total_cost: float = 0.0
        self.call_count: int = 0
        self._persist_path = persist_path
        self._lock = threading.Lock()

        # Load persisted state if exists
        if persist_path and persist_path.exists():
            self._load()

    def estimate_cost(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Estimate the cost of an API call in USD."""
        costs = MODEL_COSTS.get(model, MODEL_COSTS["default"])
        estimated = (
            (input_tokens / 1000) * costs["input"]
            + (output_tokens / 1000) * costs["output"]
        )
        return round(estimated, 6)

    def check_budget(self, estimated_cost: float) -> None:
        """
        Check if the estimated cost is within budget.

        Raises:
            BudgetExceededError: If total + estimated exceeds max.
        """
        with self._lock:
            projected = self.total_cost + estimated_cost

            # Hard stop
            if projected > self.max_cost_usd:
                raise BudgetExceededError(
                    f"Budget exceeded: projected ${projected:.4f} > "
                    f"cap ${self.max_cost_usd:.2f}. "
                    f"Already spent: ${self.total_cost:.4f}"
                )

            # Warning
            ratio = projected / self.max_cost_usd if self.max_cost_usd > 0 else 1.0
            if ratio >= self.warning_threshold:
                logger.warning(
                    f"Cost warning: at {ratio:.0%} of budget "
                    f"(${projected:.4f} / ${self.max_cost_usd:.2f})"
                )

    def record_cost(self, actual_cost: float) -> None:
        """Record the actual cost of a completed API call."""
        with self._lock:
            self.total_cost += actual_cost
            self.call_count += 1
            logger.info(
                f"Cost recorded: ${actual_cost:.6f} | "
                f"Total: ${self.total_cost:.4f} / ${self.max_cost_usd:.2f} | "
                f"Calls: {self.call_count}"
            )
            self._persist()

    def get_summary(self) -> dict:
        """Get a summary of current spending."""
        remaining = self.max_cost_usd - self.total_cost
        ratio = (
            self.total_cost / self.max_cost_usd if self.max_cost_usd > 0 else 1.0
        )
        return {
            "total_spent": round(self.total_cost, 6),
            "max_budget": self.max_cost_usd,
            "remaining": round(remaining, 6),
            "usage_ratio": round(ratio, 4),
            "call_count": self.call_count,
            "warning": ratio >= self.warning_threshold,
        }

    def reset(self) -> None:
        """Reset the cost tracker (for new tasks)."""
        with self._lock:
            self.total_cost = 0.0
            self.call_count = 0
            self._persist()

    def _persist(self) -> None:
        """Save state to disk atomically."""
        if self._persist_path:
            data = {
                "total_cost": self.total_cost,
                "call_count": self.call_count,
                "max_cost_usd": self.max_cost_usd,
            }
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._persist_path.with_suffix(".tmp")
            try:
                tmp_path.write_text(json.dumps(data, indent=2))
                os.replace(str(tmp_path), str(self._persist_path))
            except Exception as e:
                logger.error(f"Failed to persist cost state: {e}")

    def _load(self) -> None:
        """Load state from disk."""
        if self._persist_path and self._persist_path.exists():
            try:
                data = json.loads(self._persist_path.read_text())
                self.total_cost = float(data.get("total_cost", 0.0))
                self.call_count = int(data.get("call_count", 0))
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.warning(f"Corrupt cost state file, resetting: {e}")
                self.total_cost = 0.0
                self.call_count = 0
