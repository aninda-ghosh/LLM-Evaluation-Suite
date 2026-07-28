"""
GSM8K (Grade School Math 8K) Evaluator.
Handles dynamic prompt formatting and answer scoring from raw GSM8K records.
Raw record schema: {"question": str, "answer": str}
"""

import re


def format_prompt(item: dict) -> str:
    """Formats a raw GSM8K math record into an LLM user prompt dynamically at runtime."""
    question = item.get("question", "")
    return f"{question}\nSolve step-by-step and state the final numerical answer after ####."


def score_item(item: dict, clean_response: str) -> tuple:
    """
    Scores the model output against the raw GSM8K record.
    """
    raw_answer_text = item.get("answer", "")
    answer_parts = raw_answer_text.split("####")
    expected_str = answer_parts[-1].strip().replace(",", "") if len(answer_parts) > 1 else raw_answer_text.strip().replace(",", "")

    # Extract number after #### in model response
    match_hash = re.search(r"####\s*(-?\d[\d,]*\.?\d*)", clean_response)
    if match_hash:
        raw_num = match_hash.group(1).replace(",", "")
    else:
        clean_no_commas = clean_response.replace(",", "")
        numbers = re.findall(r"-?\d+(?:\.\d+)?", clean_no_commas)
        raw_num = numbers[-1] if numbers else ""

    if raw_num and expected_str:
        try:
            is_pass = abs(float(raw_num) - float(expected_str)) < 1e-3
            reason = f"Extracted number '{raw_num}', Expected '{expected_str}'"
            return is_pass, raw_num, reason
        except ValueError:
            return False, raw_num, f"Invalid float comparison ({raw_num} vs {expected_str})"

    return False, "", f"No numeric answer found. Expected '{expected_str}'"
