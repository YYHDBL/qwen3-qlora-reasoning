"""Shared raw payload and runtime datatypes for solver problems."""

import json
from pathlib import Path
from typing import Any, Literal, cast

ProblemCategory = Literal[
    "bit_manipulation",
    "cipher",
    "equation_numeric_deduce",
    "equation_numeric_guess",
    "cryptarithm_deduce",
    "cryptarithm_guess",
    "gravity",
    "numeral",
    "unit_conversion",
]


class Example:
    input_value: str
    output_value: str

    def __init__(self, input_value: str, output_value: str):
        self.input_value = input_value
        self.output_value = output_value

    def to_payload(self) -> dict[str, str]:
        return {
            "input_value": self.input_value,
            "output_value": self.output_value,
        }


class Problem:
    id: str
    category: ProblemCategory
    examples: list[Example]
    question: str
    answer: str
    prompt: str

    def __init__(
        self,
        id: str,
        category: ProblemCategory,
        examples: list[Example],
        question: str,
        answer: str,
        prompt: str = "",
    ):
        self.id = id
        self.category = category
        self.examples = examples
        self.question = question
        self.answer = answer
        self.prompt = prompt

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "Problem":
        raw_examples = cast(list[dict[str, Any]], payload["examples"])
        examples = [
            Example(
                str(example["input_value"]),
                str(example["output_value"]),
            )
            for example in raw_examples
        ]
        return cls(
            id=str(payload["id"]),
            category=cast(ProblemCategory, payload["category"]),
            examples=examples,
            question=str(payload["question"]),
            answer=str(payload["answer"]),
            prompt=str(payload.get("prompt", "")),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "id": self.id,
            "category": self.category,
            "prompt": self.prompt,
            "answer": self.answer,
            "examples": [example.to_payload() for example in self.examples],
            "question": self.question,
        }

    def to_index_payload(self) -> dict[str, str]:
        return {
            "id": self.id,
            "category": self.category,
        }

    @classmethod
    def load_from_json(cls, id: str) -> "Problem":
        with (Path("problems") / f"{id}.jsonl").open() as f:
            payload = cast(dict[str, Any], json.loads(f.readline()))
        return cls.from_payload(payload)

    @classmethod
    def load_all(cls) -> list["Problem"]:
        problems: list[Problem] = []
        for path in sorted(Path("problems").glob("*.jsonl")):
            with path.open() as f:
                line = f.readline().strip()
            if line:
                problems.append(
                    cls.from_payload(cast(dict[str, Any], json.loads(line)))
                )
        return problems


def _fmt_int_with_dp(value: int, dp: int) -> str:
    """Format an integer as a decimal string with *dp* decimal places."""
    if dp == 0:
        return str(value)
    s = str(value).zfill(dp + 1)
    s = s[: len(s) - dp] + "." + s[len(s) - dp :]
    # strip leading zeros but keep one before the dot
    s = s.lstrip("0") or "0"
    if s.startswith("."):
        s = "0" + s
    return s


def truncate_3dp(s: str) -> str:
    """Truncate a decimal string to at most 3 decimal places (no rounding)."""
    if "." not in s:
        return s
    integer, frac = s.split(".")
    if len(frac) <= 3:
        return s
    return integer + "." + frac[:3]


def _dp_count(s: str) -> int:
    if "." not in s:
        return 0
    return len(s.split(".")[1])


def pad_dp(s: str, n: int) -> str:
    """Pad a decimal string to exactly *n* decimal places."""
    if "." not in s:
        s = s + "."
    integer, frac = s.split(".")
    return integer + "." + frac.ljust(n, "0")


def cast_dp_pair(a: str, b: str) -> tuple[str, str, int, int]:
    """Pad *a* and *b* to the same number of decimal places.

    Returns (a_padded, b_padded, a_target_dp, b_target_dp).
    Target dp is the max of both; individual values show what each was cast to.
    """
    da, db = _dp_count(a), _dp_count(b)
    target = max(da, db)
    return pad_dp(a, target), pad_dp(b, target), target, target


def long_multiplication_lines(a_str: str, b_str: str) -> tuple[list[str], str]:
    """Generate step-by-step multiplication of two decimal numbers.

    Decomposes *b* into place-value components and multiplies *a* by each,
    then shows a running sum.

    Returns (lines, result_str) where result_str is the exact product.
    """
    a_dp = len(a_str.split(".")[1]) if "." in a_str else 0
    b_dp = len(b_str.split(".")[1]) if "." in b_str else 0
    total_dp = a_dp + b_dp

    a_int = int(a_str.replace(".", ""))
    b_int = int(b_str.replace(".", ""))

    lines: list[str] = []

    # Break b into place-value components, least significant digit first
    b_digits_str = str(abs(b_int))
    b_num_digits = len(b_digits_str)

    # (component_display, product_int_scaled, product_display)
    components: list[tuple[str, int, str]] = []
    for i in range(b_num_digits - 1, -1, -1):
        d = int(b_digits_str[i])
        if d == 0:
            continue
        # Component value scaled by 10^b_dp
        comp_scaled = d * (10 ** (b_num_digits - 1 - i))
        comp_display = _fmt_int_with_dp(comp_scaled, b_dp)
        if b_dp > 0:
            comp_display = pad_dp(comp_display, b_dp)

        product_int = a_int * comp_scaled  # scaled by 10^total_dp
        product_display = _fmt_int_with_dp(product_int, total_dp)
        if total_dp > 0:
            product_display = pad_dp(product_display, total_dp)

        components.append((comp_display, product_int, product_display))

    # Multiplication lines: a * component = product
    for comp_display, _, product_display in components:
        lines.append(f"{a_str} * {comp_display} = {product_display}")

    # Running sum (fold from smallest to largest)
    if len(components) >= 2:
        running = components[0][1]
        for i in range(1, len(components)):
            running_display = _fmt_int_with_dp(running, total_dp)
            if total_dp > 0:
                running_display = pad_dp(running_display, total_dp)
            running += components[i][1]
            sum_display = _fmt_int_with_dp(running, total_dp)
            if total_dp > 0:
                sum_display = pad_dp(sum_display, total_dp)
            lines.append(f"{running_display} + {components[i][2]} = {sum_display}")

    # Compute final result string
    total = a_int * b_int
    result_str = _fmt_int_with_dp(total, total_dp)
    return lines, result_str


def long_division_lines(
    numerator_str: str, denominator_str: str, max_decimal_digits: int = 3
) -> tuple[list[str], str]:
    """Generate long-division steps via repeated subtraction.

    Returns (lines, result_str) where result_str is the truncated quotient.
    """
    n_dp: int = len(numerator_str.split(".")[1]) if "." in numerator_str else 0
    d_dp: int = len(denominator_str.split(".")[1]) if "." in denominator_str else 0
    max_dp: int = max(n_dp, d_dp)

    num: int = int(round(float(numerator_str) * 10**max_dp))
    den: int = int(round(float(denominator_str) * 10**max_dp))

    lines: list[str] = []
    acc: int = 0  # accumulator as integer; real value = acc / 10^decimal_digits
    decimal_digits: int = 0

    def fmt_acc() -> str:
        if decimal_digits == 0:
            return str(acc)
        s = str(acc).zfill(decimal_digits + 1)
        return s[:-decimal_digits] + "." + s[-decimal_digits:]

    def fmt_scale() -> str:
        if decimal_digits == 0:
            return "1"
        return "0." + "0" * (decimal_digits - 1) + "1"

    def fmt_line(n: int) -> str:
        return f"= {fmt_acc()} + {fmt_scale()} * {n} / {den}"

    lines.append(fmt_line(num))

    while decimal_digits <= max_decimal_digits:
        if num >= den:
            num -= den
            acc += 1
            lines.append(fmt_line(num))
        else:
            decimal_digits += 1
            if decimal_digits > max_decimal_digits:
                break
            num *= 10
            acc *= 10
            lines.append(fmt_line(num))

    # Restore decimal_digits if it was incremented past max before breaking
    if decimal_digits > max_decimal_digits:
        decimal_digits = max_decimal_digits
    return lines, fmt_acc()
