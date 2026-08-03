"""
GSM8K (Grade School Math 8K) Evaluator Class.
Extracts numerical answers from math reasoning responses.
"""

import re
from evaluators.base import BaseEvaluator


class GSM8KEvaluator(BaseEvaluator):
    """GSM8K math dataset evaluator."""

    def format_prompt(self, item: dict) -> str:
        question = item.get("question", "")
        return f"{question}\nBe concise. Solve step-by-step and state the final numerical answer after ####."

    def get_expected_answer(self, item: dict) -> str:
        raw_answer = item.get("answer", "")
        answer_parts = raw_answer.split("####")
        return answer_parts[-1].strip().replace(",", "") if len(answer_parts) > 1 else raw_answer.strip().replace(",", "")

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
        # 1. Match #### <num>
        match_hash = re.search(r"####\s*(-?\d[\d,]*\.?\d*)", response_text)
        if match_hash:
            return match_hash.group(1).replace(",", "").rstrip(".")
        
        # 2. Match LaTeX \boxed{<num>}
        match_boxed = re.search(r"boxed\{[^0-9-]*(-?\d[\d,]*\.?\d*)", response_text)
        if match_boxed:
            return match_boxed.group(1).replace(",", "").rstrip(".")
            
        # 3. Match Final Answer: <num> / Answer: <num>
        match_ans = re.search(r"(?:final answer|answer)[^0-9-]*(-?\d[\d,]*\.?\d*)", response_text, re.IGNORECASE)
        if match_ans:
            return match_ans.group(1).replace(",", "").rstrip(".")

        # 4. Fallback: Last number in response
        numbers = re.findall(r"-?\d+(?:\.\d+)?", response_text.replace(",", ""))
        return numbers[-1].rstrip(".") if numbers else ""

    def score_item(self, item: dict, response_text: str) -> tuple[bool, str, str]:
        expected_str = self.get_expected_answer(item)
        raw_num = self.extract_answer(response_text, item)

        if raw_num and expected_str:
            try:
                is_pass = abs(float(raw_num) - float(expected_str)) < 1e-3
                return is_pass, raw_num, f"Extracted '{raw_num}', Expected '{expected_str}'"
            except ValueError:
                return False, raw_num, f"Invalid numerical comparison ('{raw_num}' vs '{expected_str}')"

        return False, raw_num, f"No numeric answer found. Expected '{expected_str}'"
