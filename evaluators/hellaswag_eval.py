"""
HellaSwag Commonsense Completion Evaluator.
Handles dynamic prompt formatting and answer scoring from raw HellaSwag records.
Raw record schema: {"ctx": str, "endings": list[str], "label": int/str}
"""

import re


def format_prompt(item: dict) -> str:
    """Formats a raw HellaSwag record into an LLM user prompt dynamically at runtime."""
    ctx = item.get("ctx", "")
    endings = item.get("endings", [])
    options_text = "\n".join([f"{chr(65+j)}) {opt}" for j, opt in enumerate(endings)])
    return f"Complete the text with the most logical ending:\nContext: {ctx}\nOptions:\n{options_text}\nAnswer with the letter (A, B, C, or D) only."


def score_item(item: dict, clean_response: str) -> tuple:
    """
    Scores the model output against the raw HellaSwag record.
    """
    raw_label = item.get("label", 0)
    label_idx = int(raw_label) if str(raw_label).isdigit() else 0
    expected = chr(65 + label_idx) if label_idx < 26 else "A"

    match_ans = re.search(
        r"(?:answer|option|choice)\s*(?:is|:)?\s*\b([A-D])\b",
        clean_response,
        flags=re.IGNORECASE,
    )
    if match_ans:
        pred = match_ans.group(1).upper()
    else:
        matches = re.findall(r"\b([A-D])\b", clean_response.upper())
        pred = matches[-1] if matches else clean_response[:1].upper()

    is_pass = (pred == expected)
    reason = f"Extracted choice '{pred}', Expected choice '{expected}'"
    return is_pass, pred, reason
