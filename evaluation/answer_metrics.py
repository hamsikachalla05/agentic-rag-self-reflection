"""Answer comparison for T²-RAGBench-style numeric and short-span answers."""

from __future__ import annotations

import ast
import re
from typing import Any


NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?")


def _try_literal(value: str) -> Any:
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def flatten_gold(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            out.extend(flatten_gold(item))
        return out
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return []
    parsed = _try_literal(text) if text[:1] in "['\"" else text
    if parsed is not text and parsed is not value:
        return flatten_gold(parsed)
    return [text]


def parse_number(text: str) -> float | None:
    raw = text.strip()
    if not raw:
        return None
    negative = raw.startswith("(") and raw.endswith(")")
    cleaned = raw.replace(",", "").replace("$", "").replace("£", "")
    cleaned = cleaned.replace("(", "").replace(")", "").strip()
    pct = cleaned.endswith("%")
    cleaned = cleaned.replace("%", "").strip()
    try:
        number = float(cleaned)
    except ValueError:
        return None
    if negative:
        number = -number
    if pct:
        number = number / 100.0
    return number


def extract_numbers(text: str) -> list[float]:
    found: list[float] = []
    for match in NUMBER_RE.finditer(text):
        number = parse_number(match.group())
        if number is not None:
            found.append(number)
    return found


def numbers_close(pred: float, gold: float) -> bool:
    if pred == gold:
        return True
    tolerance = max(1e-2, abs(gold) * 0.02)
    if abs(pred - gold) <= tolerance:
        return True
    # program_answer is often a ratio while the report prints a percent
    if abs(gold) <= 1.5 and abs(pred - gold * 100) <= max(0.5, abs(gold * 100) * 0.02):
        return True
    if abs(pred) <= 1.5 and abs(pred * 100 - gold) <= max(0.5, abs(gold) * 0.02):
        return True
    return False


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9.%]+", " ", text.lower()).strip()


def is_answer_correct(prediction: str, gold_answer: Any, original_answer: Any = None) -> bool:
    pred = (prediction or "").strip()
    if not pred:
        return False
    gold_strings = flatten_gold(gold_answer) + flatten_gold(original_answer)
    pred_norm = normalize_text(pred)
    for gold in gold_strings:
        if normalize_text(gold) and normalize_text(gold) in pred_norm:
            return True
        gold_num = parse_number(gold)
        if gold_num is None:
            continue
        for pred_num in extract_numbers(pred) or []:
            if numbers_close(pred_num, gold_num):
                return True
    return False
