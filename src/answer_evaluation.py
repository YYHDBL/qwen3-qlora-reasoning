"""Task-aware answer extraction and comparison."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any

from src.dataset_classifier import (
    BINARY_ANSWER_RE,
    GRAVITY_ANSWER_RE,
    ROMAN_RE,
    SYMBOL_TOKEN_RE,
    TASK_ORDER,
    UNIT_ANSWER_RE,
    WORDS_RE,
)


ANSWER_LABEL_RE = re.compile(
    r"^(?:final answer|answer|the answer is):[ \t]*(.*)$",
    flags=re.IGNORECASE,
)
DECIMAL_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _extract_candidate(raw_prediction: str) -> tuple[str | None, bool, str]:
    normalized_newlines = raw_prediction.replace("\r\n", "\n").replace(
        "\r", "\n"
    )
    nonempty_lines = [
        line for line in normalized_newlines.split("\n") if line.strip()
    ]
    if not nonempty_lines:
        return None, False, "empty"

    if len(nonempty_lines) == 1:
        line = nonempty_lines[0].strip()
        label_match = ANSWER_LABEL_RE.fullmatch(line)
        candidate = label_match.group(1) if label_match else line
        candidate = candidate.strip()
        return candidate or None, bool(candidate), (
            "labeled" if label_match else "plain"
        )

    labeled_candidates = []
    for index, line in enumerate(nonempty_lines):
        label_match = ANSWER_LABEL_RE.fullmatch(line.strip())
        if label_match:
            labeled_candidates.append((index, label_match.group(1).strip()))
    if len(labeled_candidates) == 1:
        _, candidate = labeled_candidates[0]
        return candidate or None, False, "labeled_with_extra_text"

    return nonempty_lines[0].strip() or None, False, "multiline"


def _parse_decimal(value: str) -> Decimal | None:
    if not DECIMAL_RE.fullmatch(value):
        return None
    try:
        return Decimal(value)
    except InvalidOperation:
        return None


def _base_result(
    raw_prediction: str,
    parsed_answer: str | None,
    parse_success: bool,
    format_valid: bool,
    extraction_method: str,
    parse_error: str | None,
) -> dict[str, Any]:
    return {
        "prediction_raw": raw_prediction,
        "parsed_answer": parsed_answer,
        "parse_success": parse_success,
        "format_valid": format_valid,
        "extraction_method": extraction_method,
        "parse_error": parse_error,
        "strict_correct": False,
        "normalized_correct": False,
        "primary_correct": False,
    }


def evaluate_answer(
    task_type: str,
    gold_raw: str,
    prediction_raw: str,
) -> dict[str, Any]:
    """Parse and compare one prediction using task-specific rules."""
    if task_type not in TASK_ORDER:
        raise ValueError(f"unsupported task type: {task_type}")

    candidate, clean_output, extraction_method = _extract_candidate(
        prediction_raw
    )
    if candidate is None:
        result = _base_result(
            prediction_raw,
            None,
            False,
            False,
            extraction_method,
            f"invalid {task_type.replace('_', ' ')} answer",
        )
        return result

    parse_success = False
    strict_format = False
    normalized_correct = False
    parse_error = None

    if task_type == "bit_manipulation":
        parse_success = BINARY_ANSWER_RE.fullmatch(candidate) is not None
        strict_format = parse_success
        normalized_correct = parse_success and candidate == gold_raw
    elif task_type == "gravity":
        predicted_decimal = _parse_decimal(candidate)
        gold_decimal = _parse_decimal(gold_raw)
        if gold_decimal is None:
            raise ValueError(f"invalid gold gravity answer: {gold_raw!r}")
        parse_success = predicted_decimal is not None
        strict_format = GRAVITY_ANSWER_RE.fullmatch(candidate) is not None
        normalized_correct = (
            parse_success and predicted_decimal == gold_decimal
        )
    elif task_type == "unit_conversion":
        predicted_decimal = _parse_decimal(candidate)
        gold_decimal = _parse_decimal(gold_raw)
        if gold_decimal is None:
            raise ValueError(
                f"invalid gold unit conversion answer: {gold_raw!r}"
            )
        parse_success = predicted_decimal is not None
        strict_format = UNIT_ANSWER_RE.fullmatch(candidate) is not None
        normalized_correct = (
            parse_success and predicted_decimal == gold_decimal
        )
    elif task_type == "numeral":
        upper_candidate = candidate.upper()
        parse_success = (
            bool(upper_candidate)
            and ROMAN_RE.fullmatch(upper_candidate) is not None
        )
        strict_format = (
            parse_success and ROMAN_RE.fullmatch(candidate) is not None
        )
        normalized_correct = (
            parse_success and upper_candidate == gold_raw.upper()
        )
    elif task_type == "cipher":
        normalized_candidate = " ".join(candidate.lower().split())
        parse_success = (
            WORDS_RE.fullmatch(normalized_candidate) is not None
        )
        strict_format = WORDS_RE.fullmatch(candidate) is not None
        normalized_gold = " ".join(gold_raw.lower().split())
        normalized_correct = (
            parse_success and normalized_candidate == normalized_gold
        )
    else:
        parse_success = (
            1 <= len(candidate) <= 4
            and SYMBOL_TOKEN_RE.fullmatch(candidate) is not None
        )
        strict_format = parse_success
        normalized_correct = parse_success and candidate == gold_raw

    if not parse_success:
        parse_error = f"invalid {task_type.replace('_', ' ')} answer"

    format_valid = clean_output and strict_format
    strict_correct = (
        parse_success
        and format_valid
        and candidate == gold_raw
    )
    if task_type == "gravity":
        primary_correct = format_valid and normalized_correct
    else:
        primary_correct = strict_correct

    result = _base_result(
        prediction_raw,
        candidate,
        parse_success,
        format_valid,
        extraction_method,
        parse_error,
    )
    result.update(
        {
            "strict_correct": strict_correct,
            "normalized_correct": normalized_correct,
            "primary_correct": primary_correct,
        }
    )
    return result
