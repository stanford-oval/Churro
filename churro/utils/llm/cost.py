"""LLM cost tracking utilities."""

from churro.utils.log_utils import logger


class LLMCostTracker:
    """Track total cost across all LLM calls."""

    def __init__(self) -> None:
        self._total_cost = 0.0

    def add_cost(self, cost: float) -> None:
        """Add cost to the total.

        Args:
            cost: Cost value in USD to add to the running total.
        """
        self._total_cost += cost

    def get_total_cost(self) -> float:
        """Get the total cost accumulated so far."""
        return self._total_cost

    def log_total_cost(self) -> None:
        """Log the total cost."""
        logger.info(f"Total LLM Cost: ${self._total_cost:.2f}")


# Global cost tracker instance
cost_tracker = LLMCostTracker()


def log_total_llm_cost() -> None:
    """Log the total LLM cost."""
    cost_tracker.log_total_cost()


def get_llm_total_cost() -> float:
    """Get the total LLM cost."""
    return cost_tracker.get_total_cost()
