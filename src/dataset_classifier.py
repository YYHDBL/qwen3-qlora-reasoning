from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence


TASK_ORDER = (
    "bit_manipulation",
    "gravity",
    "unit_conversion",
    "numeral",
    "cipher",
    "symbolic_transform",
)

TASK_PREFIXES = {
    "bit_manipulation": (
        "In Alice's Wonderland, a secret bit manipulation rule transforms "
        "8-bit binary numbers. The transformation involves operations like "
        "bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority or "
        "choice functions."
    ),
    "gravity": (
        "In Alice's Wonderland, the gravitational constant has been secretly "
        "changed. Here are some example observations:"
    ),
    "unit_conversion": (
        "In Alice's Wonderland, a secret unit conversion is applied to "
        "measurements. For example:"
    ),
    "numeral": (
        "In Alice's Wonderland, numbers are secretly converted into a "
        "different numeral system. Some examples are given below:"
    ),
    "cipher": (
        "In Alice's Wonderland, secret encryption rules are used on text. "
        "Here are some examples:"
    ),
    "symbolic_transform": (
        "In Alice's Wonderland, a secret set of transformation rules is "
        "applied to equations. Below are a few examples:"
    ),
}

NUMBER_PATTERN = r"-?\d+(?:\.\d+)?"
BINARY_EXAMPLE_RE = re.compile(r"[01]{8} -> [01]{8}")
BINARY_QUERY_RE = re.compile(r"Now, determine the output for: [01]{8}")
BINARY_ANSWER_RE = re.compile(r"[01]{8}")
GRAVITY_EXAMPLE_RE = re.compile(
    rf"For t = {NUMBER_PATTERN}s, distance = {NUMBER_PATTERN} m"
)
GRAVITY_QUERY_RE = re.compile(
    rf"Now, determine the falling distance for t = {NUMBER_PATTERN}s "
    r"given d = 0\.5\*g\*t\^2\."
)
GRAVITY_ANSWER_RE = re.compile(r"-?\d+\.\d{1,2}")
UNIT_EXAMPLE_RE = re.compile(rf"{NUMBER_PATTERN} m becomes {NUMBER_PATTERN}")
UNIT_QUERY_RE = re.compile(
    rf"Now, convert the following measurement: {NUMBER_PATTERN} m"
)
UNIT_ANSWER_RE = re.compile(r"-?\d+\.\d{2}")
NUMERAL_EXAMPLE_RE = re.compile(r"\d+ -> [IVXLCDM]+")
NUMERAL_QUERY_RE = re.compile(
    r"Now, write the number \d+ in the Wonderland numeral system\."
)
ROMAN_RE = re.compile(
    r"M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})"
    r"(?:IX|IV|V?I{0,3})"
)
WORDS_RE = re.compile(r"[a-z]+(?: [a-z]+)*")
CIPHER_EXAMPLE_RE = re.compile(
    r"[a-z]+(?: [a-z]+)* -> [a-z]+(?: [a-z]+)*"
)
CIPHER_QUERY_RE = re.compile(
    r"Now, decrypt the following text: [a-z]+(?: [a-z]+)*"
)
SYMBOL_TOKEN_RE = re.compile(r"[0-9!-/:-@\[-`{-~]+")
SYMBOL_QUERY_RE = re.compile(r"Now, determine the result for: (.+)")
ID_RE = re.compile(r"[0-9a-f]{8}")


@dataclass(frozen=True)
class Classification:
    task_type: str
    reasons: tuple[str, ...] = ()


def _fullmatch(pattern: re.Pattern[str], value: str) -> bool:
    return pattern.fullmatch(value) is not None


def _validate_example_block(
    lines: Sequence[str],
    example_pattern: re.Pattern[str],
    allowed_counts: set[int],
    header: str | None = None,
) -> list[str]:
    content_lines = [line for line in lines if line]
    reasons: list[str] = []
    example_start = 1
    if header is not None:
        if len(content_lines) < 2 or content_lines[1] != header:
            reasons.append("invalid_example_header")
        else:
            example_start = 2

    example_lines = content_lines[example_start:-1]
    if len(example_lines) not in allowed_counts:
        reasons.append(f"unexpected_example_count:{len(example_lines)}")
    invalid_count = sum(
        not _fullmatch(example_pattern, line) for line in example_lines
    )
    if invalid_count:
        reasons.append(f"invalid_example_lines:{invalid_count}")
    return reasons


def _validate_symbolic_examples(lines: Sequence[str]) -> list[str]:
    content_lines = [line for line in lines if line]
    example_lines = content_lines[1:-1]
    invalid_count = 0
    for line in example_lines:
        if " = " not in line:
            invalid_count += 1
            continue
        left, right = line.split(" = ", 1)
        if not (
            _fullmatch(SYMBOL_TOKEN_RE, left)
            and _fullmatch(SYMBOL_TOKEN_RE, right)
        ):
            invalid_count += 1

    reasons = []
    if len(example_lines) not in {3, 4, 5}:
        reasons.append(f"unexpected_example_count:{len(example_lines)}")
    if invalid_count:
        reasons.append(f"invalid_example_lines:{invalid_count}")
    return reasons


def _validate_query(
    lines: Sequence[str], query_pattern: re.Pattern[str]
) -> list[str]:
    query_lines = [line for line in lines if line.startswith("Now,")]
    if len(query_lines) != 1:
        return [f"unexpected_query_count:{len(query_lines)}"]
    if not _fullmatch(query_pattern, query_lines[0]):
        return ["invalid_query_format"]
    return []


def _is_canonical_roman(value: str) -> bool:
    return bool(value) and _fullmatch(ROMAN_RE, value)


def classify_record(prompt: str, answer: str) -> Classification:
    """Classify by fixed prefix, then cross-check structure and answer format."""
    lines = prompt.splitlines()
    if not lines:
        return Classification("unknown", ("empty_prompt",))

    prefix_matches = [
        task_type
        for task_type, prefix in TASK_PREFIXES.items()
        if lines[0] == prefix
    ]
    if len(prefix_matches) != 1:
        return Classification(
            "unknown", (f"prefix_match_count:{len(prefix_matches)}",)
        )

    task_type = prefix_matches[0]
    reasons: list[str] = []

    if task_type == "bit_manipulation":
        reasons.extend(
            _validate_example_block(
                lines,
                BINARY_EXAMPLE_RE,
                {7, 8, 9, 10},
                header="Here are some examples of input -> output:",
            )
        )
        reasons.extend(_validate_query(lines, BINARY_QUERY_RE))
        if not _fullmatch(BINARY_ANSWER_RE, answer):
            reasons.append("invalid_answer_format")
    elif task_type == "gravity":
        reasons.extend(
            _validate_example_block(lines, GRAVITY_EXAMPLE_RE, {3, 4, 5})
        )
        reasons.extend(_validate_query(lines, GRAVITY_QUERY_RE))
        if not _fullmatch(GRAVITY_ANSWER_RE, answer):
            reasons.append("invalid_answer_format")
    elif task_type == "unit_conversion":
        reasons.extend(
            _validate_example_block(lines, UNIT_EXAMPLE_RE, {3, 4, 5})
        )
        reasons.extend(_validate_query(lines, UNIT_QUERY_RE))
        if not _fullmatch(UNIT_ANSWER_RE, answer):
            reasons.append("invalid_answer_format")
    elif task_type == "numeral":
        reasons.extend(
            _validate_example_block(lines, NUMERAL_EXAMPLE_RE, {3, 4, 5})
        )
        reasons.extend(_validate_query(lines, NUMERAL_QUERY_RE))
        if not _is_canonical_roman(answer):
            reasons.append("invalid_answer_format")
    elif task_type == "cipher":
        reasons.extend(
            _validate_example_block(lines, CIPHER_EXAMPLE_RE, {3, 4, 5})
        )
        reasons.extend(_validate_query(lines, CIPHER_QUERY_RE))
        if not _fullmatch(WORDS_RE, answer):
            reasons.append("invalid_answer_format")
    elif task_type == "symbolic_transform":
        reasons.extend(_validate_symbolic_examples(lines))
        reasons.extend(_validate_query(lines, SYMBOL_QUERY_RE))
        query_lines = [line for line in lines if line.startswith("Now,")]
        if query_lines:
            query_match = SYMBOL_QUERY_RE.fullmatch(query_lines[0])
            if query_match and not _fullmatch(
                SYMBOL_TOKEN_RE, query_match.group(1)
            ):
                reasons.append("invalid_query_symbol_format")
        if not (1 <= len(answer) <= 4) or not _fullmatch(
            SYMBOL_TOKEN_RE, answer
        ):
            reasons.append("invalid_answer_format")

    if reasons:
        return Classification("unknown", tuple(sorted(set(reasons))))
    return Classification(task_type)
