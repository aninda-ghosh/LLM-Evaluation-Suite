"""
Evaluators Registry Package.
Maps benchmark names to Evaluator Class instances inheriting from BaseEvaluator.
"""

from evaluators.gsm8k_eval import GSM8KEvaluator
from evaluators.mmlu_eval import MMLUEvaluator
from evaluators.mmlu_pro_eval import MMLUProEvaluator
from evaluators.hellaswag_eval import HellaSwagEvaluator
from evaluators.truthfulqa_eval import TruthfulQAEvaluator

_EVALUATOR_CLASSES = {
    "mmlu_pro": MMLUProEvaluator,
    "mmlu": MMLUEvaluator,
    "gsm8k": GSM8KEvaluator,
    "hellaswag": HellaSwagEvaluator,
    "truthfulqa": TruthfulQAEvaluator,
}

_EVALUATORS = _EVALUATOR_CLASSES  # Alias for backward compatibility


def get_evaluator(benchmark_name: str):
    """Returns an instance of the Evaluator Class matching benchmark_name."""
    name = benchmark_name.lower().strip()
    if "mmlu_pro" in name:
        return MMLUProEvaluator()
    for key, cls in _EVALUATOR_CLASSES.items():
        if key in name:
            return cls()
    return MMLUEvaluator()
