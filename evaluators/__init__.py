"""
Evaluators Registry Package.
Provides access to evaluator modules for MMLU, GSM8K, HellaSwag, and TruthfulQA.
"""

from evaluators import mmlu_eval, gsm8k_eval, hellaswag_eval, truthfulqa_eval

_EVALUATOR_MAP = {
    "mmlu": mmlu_eval,
    "gsm8k": gsm8k_eval,
    "hellaswag": hellaswag_eval,
    "truthfulqa": truthfulqa_eval,
}


def get_evaluator(benchmark_name: str):
    """
    Returns the evaluator module for benchmark_name.
    Each evaluator module exposes:
    - format_prompt(raw_item: dict) -> str
    - score_item(raw_item: dict, clean_response: str) -> tuple[bool, str, str]
    """
    b_lower = benchmark_name.lower().strip()
    for key, module in _EVALUATOR_MAP.items():
        if key in b_lower:
            return module
    # Default fallback to MMLU evaluator
    return mmlu_eval
