"""Financial answer comparison for T²-RAGBench.

Ground-truth field
-----------------
Use ``program_answer`` as the canonical gold. It is the executed numeric
result of the dataset's (unreleased) reasoning program. ``original_answer``
is a surface form from the source dataset (e.g. ``14%`` vs ``0.1446``,
``['$216.0']`` vs ``216.0``) and is accepted as an equivalent form.

T²-RAGBench does **not** ship a program/derivation string.

Matching
--------
Predictions are parsed for numbers plus units (thousand / million / billion,
percent). Candidates are compared with relative tolerance and with clean
scale factors (1e3, 1e6, 1e9, 100, 0.01) so that table units in millions can
match a model that says "billion". Calendar years are compared as years, not
as scalable magnitudes.
"""

from __future__ import annotations

import ast
import math
import re
from typing import Any


NUMBER_RE = re.compile(
    r"(?<![\w.])([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*(%|percent|percentage)?"
    r"|(?<![\w.])\(([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\)"
)

UNIT_PATTERNS = [
    (re.compile(r"\b(trillion|tn)\b", re.I), 1e12),
    (re.compile(r"\b(billion|bn)\b", re.I), 1e9),
    (re.compile(r"\b(million|mm|mn)\b", re.I), 1e6),
    (re.compile(r"\b(thousand|thousands)\b", re.I), 1e3),
]

YES_TOKENS = {"yes", "true", "y"}
NO_TOKENS = {"no", "false", "n"}
SCALE_FACTORS = (1.0, 10.0, 100.0, 1e3, 1e6, 1e9, 1e12, 0.1, 0.01, 1e-3, 1e-6, 1e-9)


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
    if text[:1] in "['\"":
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = text
        if parsed is not text:
            return flatten_gold(parsed)
    return [text]


def parse_plain_number(text: str) -> float | None:
    raw = text.strip()
    if not raw:
        return None
    negative = raw.startswith("(") and raw.endswith(")")
    cleaned = raw.replace(",", "").replace("$", "").replace("£", "").replace("€", "")
    cleaned = cleaned.replace("(", "").replace(")", "").strip()
    pct = cleaned.endswith("%")
    cleaned = cleaned.rstrip("%").strip()
    try:
        number = float(cleaned)
    except ValueError:
        return None
    if negative:
        number = -number
    if pct:
        return number  # keep 14 from "14%"; percent scaling is handled via candidates
    return number


def _window(text: str, start: int, end: int, radius: int = 28) -> str:
    return text[max(0, start - radius) : min(len(text), end + radius)]


def extract_quantities(text: str) -> list[float]:
    """Numeric interpretations of a free-form model answer."""
    values: list[float] = []
    if not text:
        return values
    for match in NUMBER_RE.finditer(text):
        raw = match.group(1) or match.group(3)
        if raw is None:
            continue
        number = parse_plain_number(raw)
        if number is None:
            continue
        values.append(number)
        span = _window(text, match.start(), match.end())
        if match.group(2) or "%" in match.group(0) or re.search(r"\bpercent", span, re.I):
            values.append(number / 100.0)
            values.append(number * 100.0)
        for pattern, scale in UNIT_PATTERNS:
            if pattern.search(span):
                values.append(number * scale)
                break
    # whole-string parse (e.g. "94.0")
    plain = parse_plain_number(text)
    if plain is not None:
        values.append(plain)
    # unique while preserving order
    seen: set[float] = set()
    unique: list[float] = []
    for value in values:
        key = round(value, 10)
        if key not in seen:
            seen.add(key)
            unique.append(value)
    return unique


def gold_quantities(*fields: Any) -> list[float]:
    values: list[float] = []
    for field in fields:
        for text in flatten_gold(field):
            values.extend(extract_quantities(text))
            number = parse_plain_number(text)
            if number is not None:
                values.append(number)
    seen: set[float] = set()
    unique: list[float] = []
    for value in values:
        key = round(value, 10)
        if key not in seen:
            seen.add(key)
            unique.append(value)
    return unique


def _is_year(value: float) -> bool:
    return 1900.0 <= abs(value) <= 2100.0 and abs(value - round(value)) < 0.51


def numbers_equivalent(pred: float, gold: float) -> bool:
    if math.isnan(pred) or math.isnan(gold):
        return False
    if pred == gold:
        return True
    if _is_year(pred) and _is_year(gold):
        return int(round(pred)) == int(round(gold))
    denom = max(abs(pred), abs(gold), 1e-9)
    if abs(pred - gold) <= 0.05 or abs(pred - gold) / denom <= 0.02:
        return True
    if pred == 0 or gold == 0:
        return False
    if (pred > 0) != (gold > 0):
        return False
    for scale in SCALE_FACTORS:
        if scale == 1.0:
            continue
        if abs(abs(pred) - abs(gold) * scale) <= max(0.05, 0.02 * abs(gold) * scale):
            return True
    return False


def _categorical(text: str) -> str | None:
    tokens = set(re.findall(r"[a-z]+", (text or "").lower()))
    if tokens & YES_TOKENS:
        return "yes"
    if tokens & NO_TOKENS:
        return "no"
    return None


def is_answer_correct(prediction: str, gold_answer: Any, original_answer: Any = None) -> bool:
    pred = (prediction or "").strip()
    if not pred:
        return False

    gold_strings = flatten_gold(gold_answer) + flatten_gold(original_answer)
    pred_l = pred.lower()
    for gold in gold_strings:
        g = gold.lower().strip()
        if g and g in pred_l and any(ch.isalpha() for ch in g):
            return True

    pred_cat = _categorical(pred)
    for gold in gold_strings:
        gold_cat = _categorical(gold)
        if pred_cat and gold_cat and pred_cat == gold_cat:
            return True
        gold_num = parse_plain_number(gold)
        if pred_cat == "yes" and gold_num == 1.0:
            return True
        if pred_cat == "no" and gold_num == 0.0:
            return True

    pred_vals = extract_quantities(pred)
    gold_vals = gold_quantities(gold_answer, original_answer)
    for pred_v in pred_vals:
        for gold_v in gold_vals:
            if numbers_equivalent(pred_v, gold_v):
                return True
    return False


# Back-compat aliases used by older tests
def parse_number(text: str) -> float | None:
    return parse_plain_number(text)


def extract_numbers(text: str) -> list[float]:
    return extract_quantities(text)


def numbers_close(pred: float, gold: float) -> bool:
    return numbers_equivalent(pred, gold)


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9.%]+", " ", text.lower()).strip()
