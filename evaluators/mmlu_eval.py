"""
MMLU (Massive Multitask Language Understanding) Evaluator.
Handles dynamic prompt formatting and answer scoring from raw MMLU records.
Raw record schema: {"question": str, "choices": list[str], "answer": int}
"""

import re


def format_prompt(item: dict) -> str:
    """Formats a raw MMLU record into an LLM user prompt dynamically at runtime."""
    question = item.get("question", "")
    choices = item.get("choices", [])
    options_text = "\n".join([f"{chr(65+j)}) {opt}" for j, opt in enumerate(choices)])
    return f"Question: {question}\nOptions:\n{options_text}\nAnswer with the letter (A, B, C, or D) only."


def score_item(item: dict, clean_response: str) -> tuple:
    """
    Scores the model output against the raw MMLU record.

    Scoring algorithm (4-way multiple choice):
      1. Map the gold index in item["answer"] (int 0-3) to a letter
         (0->A, 1->B, 2->C, 3->D) -> `expected`.
      2. Extract the model's letter:
           a. Look for a phrase like "answer is X" / "option: X" / "choice X"
              (case-insensitive, X in A-D).
           b. Otherwise take the LAST standalone A-D letter in the response.
           c. If neither matches, use the first character of the response.
      3. Pass when pred == expected.

    Returns (is_pass, extracted_choice, reason).
    """
    raw_answer = item.get("answer", 0)
    expected = chr(65 + int(raw_answer)) if isinstance(raw_answer, int) else str(raw_answer).upper()

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
