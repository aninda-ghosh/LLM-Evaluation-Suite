"""
HellaSwag Commonsense Completion Evaluator Class.
Extracts completion option letters (A-D) with text contradiction validation.
"""

import re
from evaluators.base import BaseEvaluator


class HellaSwagEvaluator(BaseEvaluator):
    """HellaSwag context completion dataset evaluator."""

    def format_prompt(self, item: dict) -> str:
        ctx = item.get("ctx", "")
        endings = item.get("endings", [])
        options_text = "\n".join([f"{chr(65+i)}) {opt}" for i, opt in enumerate(endings)])
        return f"Complete the text with the most logical ending:\nContext: {ctx}\nOptions:\n{options_text}\nAnswer with the letter (A, B, C, or D) only."

    def get_expected_answer(self, item: dict) -> str:
        raw_label = item.get("label", 0)
        label_idx = int(raw_label) if str(raw_label).isdigit() else 0
        return chr(65 + label_idx) if label_idx < 26 else "A"

    def extract_think(self, response_text: str) -> str:
        patterns = [
            r"<\|channel\|>thought\s*(.*?)(?:<\|channel\|>|<channel\|>|$)",
            r"<think>\s*(.*?)(?:</think>|$)",
            r"<thought>\s*(.*?)(?:</thought>|$)",
            r"<reasoning>\s*(.*?)(?:</reasoning>|$)",
            r"<scratchpad>\s*(.*?)(?:</scratchpad>|$)",
        ]
        for pat in patterns:
            m = re.search(pat, response_text, flags=re.DOTALL | re.IGNORECASE)
            if m and m.group(1).strip():
                return m.group(1).strip()
        return "N/A (Direct Answer)"

    def extract_answer(self, response_text: str, item: dict = None) -> str:
        clean = re.sub(r"<(think|thought|reasoning|scratchpad)>.*?(?:</\1>|$)", "", response_text, flags=re.DOTALL | re.IGNORECASE).strip()
        match = re.search(r"(?:answer|option|choice)\s*(?:is|:)?\s*\b([A-D])\b", clean, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        matches = re.findall(r"\b([A-D])\b", clean.upper())
        return matches[-1] if matches else clean[:1].upper()

    def validate_choice_text(self, response_text: str, pred_letter: str, choices: list[str]) -> tuple[bool, str]:
        if not choices or not pred_letter:
            return True, ""
        pred_idx = ord(pred_letter.upper()) - 65
        if pred_idx < 0 or pred_idx >= len(choices):
            return True, ""

        cleaned_after = re.sub(rf"^(?:answer|option|choice)?\s*(?:is|:)?\s*{pred_letter}\s*[\)\:\-\.]*\s*", "", response_text.strip(), flags=re.IGNORECASE).strip().lower()
        if not cleaned_after:
            return True, ""

        pred_text = choices[pred_idx].strip().lower()
        words_pred = set(re.findall(r"\w+", pred_text))
        words_model = set(re.findall(r"\w+", cleaned_after))

        best_match_idx = pred_idx
        best_match_score = len(words_pred & words_model) / max(len(words_pred), 1)

        for idx, choice in enumerate(choices):
            if idx == pred_idx:
                continue
            c_text = choice.strip().lower()
            words_c = set(re.findall(r"\w+", c_text))
            score = len(words_c & words_model) / max(len(words_c), 1)
            if score > best_match_score:
                best_match_score = score
                best_match_idx = idx

        if best_match_idx != pred_idx and best_match_score > 0.6:
            other_letter = chr(65 + best_match_idx)
            return False, f"Contradiction (Selected '{pred_letter}' but text matches Option '{other_letter}')"

        return True, ""

    def score_item(self, item: dict, response_text: str) -> tuple[bool, str, str]:
        expected = self.get_expected_answer(item)
        choices = item.get("endings", [])
        pred = self.extract_answer(response_text, item)

        is_valid_text, text_reason = self.validate_choice_text(response_text, pred, choices)
        if not is_valid_text:
            return False, pred, f"Extracted '{pred}' [FAIL: {text_reason}], Expected '{expected}'"

        is_pass = (pred == expected)
        return is_pass, pred, f"Extracted '{pred}', Expected '{expected}'"
