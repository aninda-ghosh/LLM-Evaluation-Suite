"""
Abstract Base Evaluator Class Interface.
Every benchmark evaluator class must implement its own dataset-specific algorithms.
"""


class BaseEvaluator:
    """Abstract base class interface for benchmark dataset evaluators."""

    def format_prompt(self, item: dict) -> str:
        """Formats dataset item into prompt string for model generation."""
        raise NotImplementedError

    def get_expected_answer(self, item: dict) -> str:
        """Returns gold expected answer string."""
        raise NotImplementedError

    def extract_think(self, response_text: str) -> str:
        """Benchmark-specific algorithm to extract thinking / reasoning block."""
        raise NotImplementedError

    def extract_answer(self, response_text: str, item: dict = None) -> str:
        """Benchmark-specific algorithm to extract final answer from model output."""
        raise NotImplementedError

    def score_item(self, item: dict, response_text: str) -> tuple[bool, str, str]:
        """Scores response against gold expected answer. Returns (is_pass, pred_val, reason)."""
        raise NotImplementedError
