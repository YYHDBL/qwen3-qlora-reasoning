#!/usr/bin/env python3
"""Generate Stage 3 Wonderland cold-start SFT data.

This is a data builder only. It does not train, does not read Wonderland
validation/test files, and does not touch adapter directories.

The Wonderland source set is restricted by splits/wonderland_split_seed42.json:
only IDs under stage3_sft_pool may be used. Long legacy reasoner traces are
kept in debug/raw_traces.jsonl; training samples only receive compressed traces.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stage3_reasoners.store_types import Example, Problem  # noqa: E402


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
MAX_TOKEN_LENGTH = 1024

DEFAULT_INPUT_CSV = Path("data/raw/train.csv")
DEFAULT_SPLIT_PATH = Path("splits/wonderland_split_seed42.json")
DEFAULT_OUTPUT_DIR = Path("data/instruction/stage3_wonderland_cold_start")
DEFAULT_STAGE1_5_REPLAY = Path("data/instruction/stage1_5/train.jsonl")
DEFAULT_STAGE2_REPLAY = Path("data/instruction/stage2_thinking/train.jsonl")

TASK_BIT = "bit_manipulation"
TASK_NUMERAL = "numeral"
TASK_UNIT = "unit_conversion"
TASK_GRAVITY = "gravity"
TASK_CIPHER = "cipher"
TASK_SYMBOLIC = "symbolic_equation"
WONDERLAND_TASK_TYPES = {
    TASK_BIT,
    TASK_NUMERAL,
    TASK_UNIT,
    TASK_GRAVITY,
    TASK_CIPHER,
    TASK_SYMBOLIC,
}

SAMPLE_ANSWER_ONLY = "wonderland_answer_only"
SAMPLE_COMPRESSED_COT = "wonderland_compressed_cot"
SAMPLE_STAGE1_5_REPLAY = "stage1_5_strict_replay"
SAMPLE_STAGE2_REPLAY = "stage2_thinking_replay"
REPLAY_SAMPLE_TYPES = {SAMPLE_STAGE1_5_REPLAY, SAMPLE_STAGE2_REPLAY}

BINARY_ANSWER_RE = re.compile(r"[01]{8}")
ROMAN_ANSWER_RE = re.compile(r"[IVXLCDM]+")
NUMERIC_ANSWER_RE = re.compile(r"-?\d+(?:\.\d+)?")
BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
EQUATION_NUMERIC_RE = re.compile(r"^\d+\D\d+$")
NUMBER_PATTERN = r"-?\d+(?:\.\d+)?"

BIT_PREFIX = "In Alice's Wonderland, a secret bit manipulation rule transforms"
GRAVITY_PREFIX = "In Alice's Wonderland, the gravitational constant has been secretly changed"
UNIT_PREFIX = "In Alice's Wonderland, a secret unit conversion is applied"
CIPHER_PREFIX = "In Alice's Wonderland, secret encryption rules are used on text"
NUMERAL_PREFIX = "In Alice's Wonderland, numbers are secretly converted"
SYMBOLIC_PREFIX = "In Alice's Wonderland, a secret set of transformation rules is applied to equations"

TASK_PREFIXES = {
    TASK_BIT: BIT_PREFIX,
    TASK_GRAVITY: GRAVITY_PREFIX,
    TASK_UNIT: UNIT_PREFIX,
    TASK_CIPHER: CIPHER_PREFIX,
    TASK_NUMERAL: NUMERAL_PREFIX,
    TASK_SYMBOLIC: SYMBOLIC_PREFIX,
}

TOKENIZER_CANDIDATES = (
    "models/Qwen3-1.7B-Base",
    "Qwen/Qwen3-1.7B-Base",
)


@dataclass(frozen=True)
class ReasonerResult:
    task_type: str
    answer: str
    compressed_trace: str
    raw_trace: str
    ok: bool
    error: str = ""
    confidence: str = "low"
    metadata: Mapping[str, Any] | None = None


class QwenTokenCounter:
    """Token counter backed by a Qwen3 tokenizer, with no heuristic fallback."""

    def __init__(
        self,
        tokenizer: Any | None = None,
        *,
        model_id: str | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self.tokenizer = tokenizer or self._load_tokenizer(model_id, cache_dir)
        self.name = str(getattr(self.tokenizer, "name_or_path", model_id or "injected-tokenizer"))
        self.kind = "real"

    @staticmethod
    def _load_tokenizer(model_id: str | None, cache_dir: Path | None) -> Any:
        try:
            from transformers import AutoTokenizer
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Qwen3 tokenizer is required for Stage 3 token length checks. "
                "Install transformers or pass an injected tokenizer in tests."
            ) from exc

        candidates = (model_id,) if model_id else TOKENIZER_CANDIDATES
        last_exc: Exception | None = None
        for candidate in candidates:
            if not candidate:
                continue
            try:
                return AutoTokenizer.from_pretrained(
                    candidate,
                    cache_dir=str(cache_dir) if cache_dir else None,
                    local_files_only=str(candidate).startswith("models/"),
                    trust_remote_code=True,
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
        raise RuntimeError(f"failed to load Qwen3 tokenizer: {last_exc}") from last_exc

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        if hasattr(self.tokenizer, "encode"):
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        tokenized = self.tokenizer(text, add_special_tokens=False)
        return self._extract_token_count(tokenized)

    def count_messages(self, messages: Sequence[Mapping[str, str]]) -> int:
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                token_ids = self.tokenizer.apply_chat_template(
                    list(messages),
                    tokenize=True,
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
            except TypeError:
                token_ids = self.tokenizer.apply_chat_template(
                    list(messages),
                    tokenize=True,
                    add_generation_prompt=False,
                )
            return self._extract_token_count(token_ids)
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        return self.count_text(text)

    @staticmethod
    def _extract_token_count(token_ids: Any) -> int:
        if isinstance(token_ids, list):
            return len(token_ids)
        if hasattr(token_ids, "input_ids"):
            return len(token_ids.input_ids)
        if hasattr(token_ids, "__len__"):
            return len(token_ids)
        return 0


def log(message: str) -> None:
    timestamp = datetime.now().astimezone().strftime("%H:%M:%S")
    print(f"[{timestamp}] [stage3] {message}", file=sys.stderr, flush=True)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def prompt_hash(text: str) -> str:
    return hashlib.sha256(normalize_whitespace(text).encode("utf-8")).hexdigest()


def make_messages(user: str, assistant: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def get_user_and_assistant(messages: Any) -> tuple[str, str]:
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    user = next(
        (m.get("content") for m in messages if isinstance(m, Mapping) and m.get("role") == "user"),
        None,
    )
    assistant = next(
        (
            m.get("content")
            for m in reversed(messages)
            if isinstance(m, Mapping) and m.get("role") == "assistant"
        ),
        None,
    )
    if not isinstance(user, str) or not isinstance(assistant, str):
        raise ValueError("messages must contain string user and assistant content")
    return user, assistant


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} invalid JSON: {exc}") from exc
    return rows


def detect_task_type(prompt: str) -> str:
    first_line = prompt.splitlines()[0] if prompt.splitlines() else ""
    for task_type, prefix in TASK_PREFIXES.items():
        if first_line.startswith(prefix):
            return task_type
    raise ValueError("unknown_task_type")


def _parse_arrow_examples(prompt: str) -> list[Example]:
    examples: list[Example] = []
    for line in prompt.splitlines():
        if " -> " not in line:
            continue
        left, right = line.split(" -> ", 1)
        examples.append(Example(left.strip(), right.strip()))
    return examples


def _parse_symbolic_examples(prompt: str) -> list[Example]:
    examples: list[Example] = []
    for line in prompt.splitlines():
        if " = " not in line or line.startswith("Now,"):
            continue
        left, right = line.split(" = ", 1)
        examples.append(Example(left.strip(), right.strip()))
    return examples


def _problem_category(task_type: str, question: str) -> str:
    if task_type == TASK_SYMBOLIC:
        return "equation_numeric_deduce" if EQUATION_NUMERIC_RE.fullmatch(question) else "cryptarithm_deduce"
    return task_type


def parse_wonderland_problem(row: Mapping[str, str]) -> Problem:
    pid = str(row["id"])
    prompt = str(row["prompt"])
    answer = str(row["answer"]).strip()
    task_type = detect_task_type(prompt)

    if task_type == TASK_BIT:
        examples = [
            Example(a, b)
            for a, b in re.findall(r"([01]{8}) -> ([01]{8})", prompt)
        ]
        match = re.search(r"Now, determine the output for: ([01]{8})", prompt)
    elif task_type == TASK_NUMERAL:
        examples = _parse_arrow_examples(prompt)
        match = re.search(r"write the number (\d+) in the Wonderland numeral system", prompt)
    elif task_type == TASK_UNIT:
        examples = [
            Example(a, b)
            for a, b in re.findall(
                rf"({NUMBER_PATTERN})\s+\S+\s+becomes\s+({NUMBER_PATTERN})",
                prompt,
            )
        ]
        match = re.search(rf"convert the following measurement:\s+({NUMBER_PATTERN})\s+\S+", prompt)
    elif task_type == TASK_GRAVITY:
        examples = [
            Example(t, d)
            for t, d in re.findall(
                rf"For t = ({NUMBER_PATTERN})s, distance = ({NUMBER_PATTERN}) m",
                prompt,
            )
        ]
        match = re.search(rf"falling distance for t = ({NUMBER_PATTERN})s", prompt)
    elif task_type == TASK_CIPHER:
        examples = _parse_arrow_examples(prompt)
        match = re.search(r"decrypt the following text: (.+)$", prompt)
    else:
        examples = _parse_symbolic_examples(prompt)
        match = re.search(r"determine the result for: (.+)$", prompt)

    if match is None:
        raise ValueError(f"{task_type}:missing_question")
    question = match.group(1).strip()
    if not examples:
        raise ValueError(f"{task_type}:missing_examples")
    return Problem(
        id=pid,
        category=_problem_category(task_type, question),
        examples=examples,
        question=question,
        answer=answer,
        prompt=prompt,
    )


def task_type_for_problem(problem: Problem) -> str:
    if problem.category in {"equation_numeric_deduce", "equation_numeric_guess", "cryptarithm_deduce", "cryptarithm_guess"}:
        return TASK_SYMBOLIC
    return str(problem.category)


def _extract_reasoner_answer(raw_trace: str) -> str:
    matches = [m.strip() for m in BOXED_RE.findall(raw_trace)]
    matches = [m for m in matches if m and m != "–"]
    return matches[-1] if matches else ""


def _extract_selected_rules(raw_trace: str) -> tuple[str, ...]:
    lines = raw_trace.splitlines()
    try:
        start = lines.index("Selected") + 1
    except ValueError:
        return ()
    selected: list[str] = []
    for line in lines[start:]:
        stripped = line.strip()
        if not stripped:
            break
        parts = stripped.split(maxsplit=1)
        if len(parts) != 2 or not parts[0].isdigit():
            break
        selected.append(parts[1])
    return tuple(selected)


def _clean_trace_line(text: str) -> str:
    text = re.sub(r"\\boxed\{[^}]*\}", "", text)
    return normalize_whitespace(text)


def build_bit_trace(problem: Problem, answer: str, raw_trace: str) -> str:
    rules = _extract_selected_rules(raw_trace)
    rule_text = ", ".join(f"b{i}:{rule}" for i, rule in enumerate(rules)) or "rule vector inferred from examples"
    return "\n".join(
        [
            "Task type: bit_manipulation over 8-bit binary strings.",
            f"Selected 8-bit rule: {rule_text}.",
            f"Apply that rule bit-by-bit to query {problem.question}.",
            f"The final 8-bit output is {answer}.",
        ]
    )


def build_numeral_trace(problem: Problem, answer: str, raw_trace: str) -> str:
    del raw_trace
    return "\n".join(
        [
            "Task type: numeral conversion.",
            "Use the Roman place-value rules inferred from the examples.",
            f"Convert {problem.question} greedily from largest symbol to smallest.",
            f"The Wonderland numeral is {answer}.",
        ]
    )


def _format_compact_number(value: float) -> str:
    text = f"{value:.6g}"
    return "0" if text == "-0" else text


def _numeric_close(predicted: float, gold: float) -> bool:
    abs_error = abs(predicted - gold)
    if abs_error <= 0.05:
        return True
    denom = max(abs(gold), 1e-12)
    return abs_error / denom <= 0.005


def _parse_float(value: str) -> float:
    return float(value.strip())


def _coefficient(values: Sequence[float]) -> float:
    return float(statistics.median(values))


def build_unit_trace(coef: float) -> str:
    coef_text = _format_compact_number(coef)
    return "\n".join(
        [
            "Task type: unit conversion.",
            f"Compute output/input ratios from the examples. The ratios are consistent around {coef_text}.",
            "Apply the coefficient to the query value and round to the required format.",
        ]
    )


def build_gravity_trace(g_value: float) -> str:
    g_text = _format_compact_number(g_value)
    return "\n".join(
        [
            "Task type: gravity.",
            f"Use g = 2*d/t^2 from the examples. The examples give a consistent g around {g_text}.",
            "Then compute d = 0.5*g*t^2 for the query time and round to the required format.",
        ]
    )


def build_cipher_trace(problem: Problem, answer: str, raw_trace: str) -> str:
    del raw_trace
    return "\n".join(
        [
            "Task type: cipher.",
            "Infer the substitution mapping from encrypted/plaintext examples.",
            f"Apply the mapping to query text: {problem.question}.",
            f"The decrypted text is {answer}.",
        ]
    )


def build_symbolic_trace(problem: Problem, answer: str, raw_trace: str) -> str:
    """Build a compressed CoT trace for symbolic equation problems.

    Returns empty string when the reasoner cannot provide a specific rule,
    signalling the caller to degrade to answer-only (no compressed_cot row).
    """
    rule_text = _extract_symbolic_rule(raw_trace)
    if not rule_text:
        return ""
    return "\n".join(
        [
            "Task type: symbolic/equation-like transformation.",
            f"Rule: {rule_text}.",
            f"Query: {problem.question}",
            f"Result: {answer}",
        ]
    )


def _extract_symbolic_rule(raw_trace: str) -> str:
    """Extract the specific rule applied from a symbolic reasoner trace.

    Returns empty string if the rule is too generic or unknown.
    """
    for line in raw_trace.splitlines():
        s = line.strip()
        # equation_numeric: "match, correct, actions: reversed operands, reversed result, addition"
        m = re.search(r"match,\s*correct,\s*actions:\s*(.+)", s)
        if m:
            return m.group(1).rstrip(",")
    if "The question operator is found in the examples." in raw_trace:
        return ""
    for line in raw_trace.splitlines():
        s = line.strip()
        # cryptarithm: "The question operator is 【+】, which is concatenation."
        m = re.search(r"The question operator is (.+?), which is (.+?)\.?$", s)
        if m:
            operator = m.group(1).strip()
            rule = m.group(2).strip().rstrip(".")
            if rule.lower() == "unknown":
                return ""
            return f"{rule} on operator {operator}"
    return ""


TraceBuilder = Callable[[Problem, str, str], str]
ReasonerFn = Callable[[Problem], str | None]


def _result_from_legacy(
    problem: Problem,
    task_type: str,
    fn: ReasonerFn,
    builder: TraceBuilder,
) -> ReasonerResult:
    try:
        raw_trace = fn(problem)
    except Exception as exc:  # noqa: BLE001
        return ReasonerResult(task_type, "", "", "", False, f"reasoner_error:{exc}", "low")
    if not raw_trace:
        return ReasonerResult(task_type, "", "", "", False, "reasoner_returned_none", "low")
    answer = _extract_reasoner_answer(raw_trace).strip()
    if not answer:
        return ReasonerResult(task_type, "", "", raw_trace, False, "missing_reasoner_answer", "low")
    compressed = builder(problem, answer, raw_trace)
    if not compressed:
        return ReasonerResult(task_type, answer, "", raw_trace, False, "no_specific_rule_for_cot", "low")
    if "\\boxed" in compressed:
        return ReasonerResult(task_type, answer, "", raw_trace, False, "compressed_trace_boxed_residue", "low")
    return ReasonerResult(task_type, answer, compressed, raw_trace, True, "", "high")


def reason_bit_manipulation(problem: Problem) -> ReasonerResult:
    from stage3_reasoners.bit_manipulation import reasoning_bit_manipulation

    return _result_from_legacy(problem, TASK_BIT, reasoning_bit_manipulation, build_bit_trace)


def reason_numeral(problem: Problem) -> ReasonerResult:
    from stage3_reasoners.numeral import reasoning_numeral

    return _result_from_legacy(problem, TASK_NUMERAL, reasoning_numeral, build_numeral_trace)


def reason_unit_conversion(problem: Problem) -> ReasonerResult:
    ratios: list[float] = []
    raw_lines = ["local deterministic unit_conversion reasoner"]
    try:
        for ex in problem.examples:
            inp = _parse_float(ex.input_value)
            out = _parse_float(ex.output_value)
            if inp == 0:
                continue
            ratio = out / inp
            ratios.append(ratio)
            raw_lines.append(f"example input={ex.input_value} output={ex.output_value} ratio={ratio:.12g}")
        if not ratios:
            return ReasonerResult(TASK_UNIT, "", "", "\n".join(raw_lines), False, "no_nonzero_examples", "low")
        coef = _coefficient(ratios)
        query_value = _parse_float(problem.question)
        predicted = query_value * coef
        gold = _parse_float(problem.answer)
    except Exception as exc:  # noqa: BLE001
        return ReasonerResult(TASK_UNIT, "", "", "\n".join(raw_lines), False, f"parse_error:{exc}", "low")

    raw_lines.extend(
        [
            f"coefficient={coef:.12g}",
            f"query_value={problem.question}",
            f"predicted={predicted:.12g}",
            f"gold={problem.answer}",
        ]
    )
    if not _numeric_close(predicted, gold):
        raw_lines.append(f"numeric_tolerance_failed abs_error={abs(predicted - gold):.12g}")
        return ReasonerResult(TASK_UNIT, "", "", "\n".join(raw_lines), False, "numeric_tolerance_failed", "low")
    compressed = build_unit_trace(coef)
    return ReasonerResult(
        TASK_UNIT,
        problem.answer,
        compressed,
        "\n".join(raw_lines),
        True,
        "",
        "high",
        {"coefficient": coef, "predicted": predicted},
    )


def reason_gravity(problem: Problem) -> ReasonerResult:
    g_values: list[float] = []
    raw_lines = ["local deterministic gravity reasoner"]
    try:
        for ex in problem.examples:
            t = _parse_float(ex.input_value)
            distance = _parse_float(ex.output_value)
            if t == 0:
                continue
            g = 2 * distance / (t * t)
            g_values.append(g)
            raw_lines.append(f"example t={ex.input_value} distance={ex.output_value} g={g:.12g}")
        if not g_values:
            return ReasonerResult(TASK_GRAVITY, "", "", "\n".join(raw_lines), False, "no_nonzero_examples", "low")
        g_value = _coefficient(g_values)
        query_t = _parse_float(problem.question)
        predicted = 0.5 * g_value * query_t * query_t
        gold = _parse_float(problem.answer)
    except Exception as exc:  # noqa: BLE001
        return ReasonerResult(TASK_GRAVITY, "", "", "\n".join(raw_lines), False, f"parse_error:{exc}", "low")

    raw_lines.extend(
        [
            f"g={g_value:.12g}",
            f"query_time={problem.question}",
            f"predicted={predicted:.12g}",
            f"gold={problem.answer}",
        ]
    )
    if not _numeric_close(predicted, gold):
        raw_lines.append(f"numeric_tolerance_failed abs_error={abs(predicted - gold):.12g}")
        return ReasonerResult(TASK_GRAVITY, "", "", "\n".join(raw_lines), False, "numeric_tolerance_failed", "low")
    compressed = build_gravity_trace(g_value)
    return ReasonerResult(
        TASK_GRAVITY,
        problem.answer,
        compressed,
        "\n".join(raw_lines),
        True,
        "",
        "high",
        {"g": g_value, "predicted": predicted},
    )


def reason_cipher(problem: Problem) -> ReasonerResult:
    from stage3_reasoners.cipher import reasoning_cipher

    return _result_from_legacy(problem, TASK_CIPHER, reasoning_cipher, build_cipher_trace)


def reason_symbolic_equation(problem: Problem) -> ReasonerResult:
    from stage3_reasoners.cryptarithm import reasoning_cryptarithm
    from stage3_reasoners.equation_numeric import reasoning_equation_numeric

    if EQUATION_NUMERIC_RE.fullmatch(problem.question):
        attempts = (reasoning_equation_numeric, reasoning_cryptarithm)
    else:
        attempts = (reasoning_cryptarithm, reasoning_equation_numeric)
    errors: list[str] = []
    for fn in attempts:
        result = _result_from_legacy(problem, TASK_SYMBOLIC, fn, build_symbolic_trace)
        if result.ok:
            return result
        errors.append(result.error)
    return ReasonerResult(TASK_SYMBOLIC, "", "", "", False, "symbolic_reasoners_failed:" + ";".join(errors), "low")


class ReasonerRegistry:
    def __init__(self, reasoners: Mapping[str, Callable[[Problem], ReasonerResult]]) -> None:
        self._reasoners = dict(reasoners)

    def run(self, problem: Problem) -> ReasonerResult:
        task_type = task_type_for_problem(problem)
        fn = self._reasoners.get(task_type)
        if fn is None:
            return ReasonerResult(task_type, "", "", "", False, "missing_reasoner", "low")
        return fn(problem)


def build_reasoner_registry() -> ReasonerRegistry:
    return ReasonerRegistry(
        {
            TASK_BIT: reason_bit_manipulation,
            TASK_NUMERAL: reason_numeral,
            TASK_UNIT: reason_unit_conversion,
            TASK_GRAVITY: reason_gravity,
            TASK_CIPHER: reason_cipher,
            TASK_SYMBOLIC: reason_symbolic_equation,
        }
    )


def build_answer_only_prompt(problem: Problem) -> str:
    return problem.prompt.rstrip() + "\n\nReturn only the final answer. Do not explain."


def build_cot_prompt(problem: Problem) -> str:
    return (
        problem.prompt.rstrip()
        + f"\n\nThink briefly in {THINK_OPEN}...{THINK_CLOSE}, then output only the final answer."
    )


def build_thinking_completion(compressed_trace: str, final_answer: str) -> str:
    return f"{THINK_OPEN}\n{compressed_trace.strip()}\n{THINK_CLOSE}\n\n{final_answer.strip()}"


def parse_thinking(content: str) -> tuple[str, str] | None:
    if content.count(THINK_OPEN) != 1 or content.count(THINK_CLOSE) != 1:
        return None
    open_idx = content.find(THINK_OPEN)
    close_idx = content.find(THINK_CLOSE)
    if open_idx >= close_idx:
        return None
    reasoning = content[open_idx + len(THINK_OPEN) : close_idx].strip()
    final = content[close_idx + len(THINK_CLOSE) :].strip()
    if not reasoning or not final:
        return None
    return reasoning, final


def parse_final_answer(completion: str) -> str:
    if THINK_OPEN in completion or THINK_CLOSE in completion:
        parsed = parse_thinking(completion)
        return "" if parsed is None else parsed[1]
    return completion.strip()


def final_answer_parse_ok(task_type: str, answer: str) -> bool:
    if not answer or "\n" in answer or "<|im_end|>" in answer or "\\boxed" in answer:
        return False
    if task_type == TASK_BIT:
        return BINARY_ANSWER_RE.fullmatch(answer) is not None
    if task_type == TASK_NUMERAL:
        return ROMAN_ANSWER_RE.fullmatch(answer) is not None
    if task_type in {TASK_UNIT, TASK_GRAVITY}:
        return NUMERIC_ANSWER_RE.fullmatch(answer) is not None
    if task_type == TASK_CIPHER:
        return bool(re.fullmatch(r"[a-z]+(?: [a-z]+)*", answer))
    if task_type == TASK_SYMBOLIC:
        return bool(answer.strip())
    return bool(answer.strip())


def _base_wonderland_record(
    *,
    record_id: str,
    problem: Problem,
    sample_type: str,
    user: str,
    assistant: str,
    token_counter: QwenTokenCounter,
    reasoner_result: ReasonerResult | None,
) -> dict[str, Any]:
    messages = make_messages(user, assistant)
    final_answer = parse_final_answer(assistant)
    return {
        "id": record_id,
        "task_type": task_type_for_problem(problem),
        "sample_type": sample_type,
        "messages": messages,
        "metadata": {
            "source": "wonderland_train",
            "source_id": problem.id,
            "source_split": "stage3_sft_pool",
            "source_prompt_hash": prompt_hash(problem.prompt),
            "prompt_hash": prompt_hash(user),
            "gold_answer": problem.answer,
            "final_answer": final_answer,
            "token_length": token_counter.count_messages(messages),
            "token_length_method": "qwen3_tokenizer",
            "reasoner": {
                "ok": bool(reasoner_result.ok) if reasoner_result else False,
                "error": reasoner_result.error if reasoner_result else "",
                "confidence": reasoner_result.confidence if reasoner_result else "",
                "answer": reasoner_result.answer if reasoner_result else "",
            },
        },
    }


def make_answer_only_record(problem: Problem, index: int, token_counter: QwenTokenCounter) -> dict[str, Any]:
    return _base_wonderland_record(
        record_id=f"stage3-wonderland-answer_only-{index:06d}-{problem.id}",
        problem=problem,
        sample_type=SAMPLE_ANSWER_ONLY,
        user=build_answer_only_prompt(problem),
        assistant=problem.answer,
        token_counter=token_counter,
        reasoner_result=None,
    )


def make_cot_record(
    problem: Problem,
    result: ReasonerResult,
    index: int,
    token_counter: QwenTokenCounter,
) -> dict[str, Any]:
    return _base_wonderland_record(
        record_id=f"stage3-wonderland-compressed_cot-{index:06d}-{problem.id}",
        problem=problem,
        sample_type=SAMPLE_COMPRESSED_COT,
        user=build_cot_prompt(problem),
        assistant=build_thinking_completion(result.compressed_trace, result.answer),
        token_counter=token_counter,
        reasoner_result=result,
    )


def make_replay_record(
    row: Mapping[str, Any],
    *,
    sample_type: str,
    split: str,
    index: int,
    token_counter: QwenTokenCounter,
) -> dict[str, Any]:
    user, assistant = get_user_and_assistant(row.get("messages"))
    messages = make_messages(user, assistant.strip())
    final = parse_final_answer(messages[1]["content"])
    return {
        "id": f"stage3-{split}-{sample_type}-{index:06d}",
        "task_type": sample_type,
        "sample_type": sample_type,
        "messages": messages,
        "metadata": {
            "source": sample_type,
            "source_id": str(row.get("id") or ""),
            "source_split": split,
            "source_category": str(row.get("category") or row.get("sample_type") or ""),
            "prompt_hash": prompt_hash(user),
            "final_answer": final,
            "token_length": token_counter.count_messages(messages),
            "token_length_method": "qwen3_tokenizer",
        },
    }


def validate_stage3_record(row: Mapping[str, Any]) -> None:
    for field in ("id", "task_type", "sample_type", "messages", "metadata"):
        if field not in row:
            raise ValueError(f"missing field: {field}")
    messages = row["messages"]
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError("messages must contain exactly user and assistant")
    if messages[0] != {"role": "user", "content": messages[0].get("content", "")}:
        raise ValueError("first message must be user/content")
    if messages[1] != {"role": "assistant", "content": messages[1].get("content", "")}:
        raise ValueError("second message must be assistant/content")
    user = messages[0]["content"]
    assistant = messages[1]["content"]
    meta = row["metadata"]
    sample_type = str(row["sample_type"])
    task_type = str(row["task_type"])
    if not isinstance(user, str) or not user.strip():
        raise ValueError("empty user content")
    if not isinstance(assistant, str) or not assistant.strip():
        raise ValueError("empty assistant content")
    if "<|im_end|>" in assistant:
        raise ValueError("assistant must not contain <|im_end|>")
    if "\\boxed" in assistant:
        raise ValueError("assistant must not contain boxed residue")
    if int(meta.get("token_length", 0)) > MAX_TOKEN_LENGTH:
        raise ValueError(f"token length exceeds {MAX_TOKEN_LENGTH}")
    if sample_type == SAMPLE_COMPRESSED_COT:
        parsed = parse_thinking(assistant)
        if parsed is None:
            raise ValueError("compressed CoT sample missing closed think tags")
        reasoning, final = parsed
        if THINK_OPEN in final or THINK_CLOSE in final:
            raise ValueError("think tags after close are not allowed")
        if "\n" in final:
            raise ValueError("after </think> there must be only final answer")
        if reasoning.count("\n") > 8:
            raise ValueError("compressed trace is too long")
    elif sample_type == SAMPLE_ANSWER_ONLY:
        if THINK_OPEN in assistant or THINK_CLOSE in assistant:
            raise ValueError("answer-only sample must not contain think tags")
    elif sample_type in REPLAY_SAMPLE_TYPES:
        if THINK_OPEN in assistant or THINK_CLOSE in assistant:
            if parse_thinking(assistant) is None:
                raise ValueError("replay thinking sample has invalid think tags")
        return
    else:
        raise ValueError(f"unsupported sample_type: {sample_type!r}")

    if task_type not in WONDERLAND_TASK_TYPES:
        raise ValueError(f"unsupported task_type: {task_type!r}")
    final = parse_final_answer(assistant)
    if not final_answer_parse_ok(task_type, final):
        raise ValueError(f"final answer parse failed: {final!r}")
    if final != meta.get("gold_answer"):
        raise ValueError(f"answer != gold: {final!r} != {meta.get('gold_answer')!r}")
    if final != meta.get("final_answer"):
        raise ValueError("metadata final_answer mismatch")
    if sample_type == SAMPLE_COMPRESSED_COT:
        reasoner = meta.get("reasoner", {})
        if isinstance(reasoner, Mapping) and reasoner.get("answer") != final:
            raise ValueError("reasoner answer != final answer")


def _read_train_csv_pool(path: Path, pool_ids: set[str]) -> tuple[list[dict[str, str]], set[str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["id", "prompt", "answer"]:
            raise ValueError(f"unexpected CSV fields: {reader.fieldnames}")
        for row in reader:
            rid = row["id"]
            if rid not in pool_ids:
                continue
            seen.add(rid)
            rows.append({"id": rid, "prompt": row["prompt"], "answer": row["answer"]})
    return rows, pool_ids - seen


def read_stage3_sft_pool(split_path: Path) -> list[str]:
    if not split_path.exists():
        raise FileNotFoundError(f"wonderland split file not found: {split_path}")
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        pool = payload
    elif isinstance(payload, Mapping):
        pool = payload.get("stage3_sft_pool")
    else:
        raise ValueError("wonderland split must be a JSON object or list")
    if not isinstance(pool, list) or not all(isinstance(item, str) for item in pool):
        raise ValueError("wonderland split stage3_sft_pool must be a list of ids")
    if not pool:
        raise ValueError("wonderland split stage3_sft_pool is empty")
    return list(dict.fromkeys(pool))


def select_wonderland_problems(
    *,
    input_csv: Path,
    split_path: Path,
    max_stage3_prompts: int,
    seed: int,
    skipped: Counter[str],
) -> tuple[list[Problem], set[str], set[str]]:
    pool_ids_list = read_stage3_sft_pool(split_path)
    pool_ids = set(pool_ids_list)
    raw_rows, missing = _read_train_csv_pool(input_csv, pool_ids)
    for _ in missing:
        skipped["split_id_missing_from_train_csv"] += 1

    parsed: list[Problem] = []
    seen_source_hashes: set[str] = set()
    for row in raw_rows:
        try:
            problem = parse_wonderland_problem(row)
        except ValueError as exc:
            skipped[f"parse_{exc}"] += 1
            continue
        source_hash = prompt_hash(problem.prompt)
        if source_hash in seen_source_hashes:
            skipped["duplicate_source_prompt_hash"] += 1
            continue
        seen_source_hashes.add(source_hash)
        parsed.append(problem)

    rng = random.Random(seed)
    rng.shuffle(parsed)
    if max_stage3_prompts > 0:
        skipped["reserved_for_rl_from_stage3_sft_pool"] += max(0, len(parsed) - max_stage3_prompts)
        parsed = parsed[:max_stage3_prompts]
    return parsed, pool_ids, missing


def _partition_train_dev(items: Sequence[Any], dev_ratio: float) -> tuple[list[Any], list[Any]]:
    if not 0 <= dev_ratio < 1:
        raise ValueError("--dev-ratio must be in [0, 1)")
    values = list(items)
    if not values:
        return [], []
    dev_count = int(round(len(values) * dev_ratio))
    if dev_ratio > 0 and dev_count == 0 and len(values) > 1:
        dev_count = 1
    dev = values[:dev_count]
    train = values[dev_count:]
    if not train and dev:
        train, dev = dev[:1], dev[1:]
    return train, dev


def replay_candidate_ok(row: Mapping[str, Any], sample_type: str) -> bool:
    try:
        user, assistant = get_user_and_assistant(row.get("messages"))
    except ValueError:
        return False
    if not user.strip() or not assistant.strip():
        return False
    if "<|im_end|>" in assistant or "\\boxed" in assistant:
        return False
    if sample_type == SAMPLE_STAGE1_5_REPLAY and (THINK_OPEN in assistant or THINK_CLOSE in assistant):
        return False
    if sample_type == SAMPLE_STAGE2_REPLAY and (THINK_OPEN in assistant or THINK_CLOSE in assistant):
        return parse_thinking(assistant) is not None
    return True


def load_replay_rows(
    path: Path,
    *,
    sample_type: str,
    skipped: Counter[str],
) -> list[dict[str, Any]]:
    if not path.exists():
        skipped[f"{sample_type}_missing"] += 1
        return []
    out: list[dict[str, Any]] = []
    for row in read_jsonl(path):
        if replay_candidate_ok(row, sample_type):
            out.append(row)
        else:
            skipped[f"{sample_type}_rejected"] += 1
    return out


def _append_if_valid(
    rows: list[dict[str, Any]],
    row: dict[str, Any],
    *,
    skipped: Counter[str],
    seen_prompt_hashes: set[str],
) -> None:
    try:
        validate_stage3_record(row)
    except ValueError as exc:
        skipped[f"validation_{exc}"] += 1
        return
    h = str(row["metadata"]["prompt_hash"])
    if h in seen_prompt_hashes:
        skipped["duplicate_prompt_hash"] += 1
        return
    seen_prompt_hashes.add(h)
    rows.append(row)


def build_wonderland_rows_for_split(
    problems: Sequence[Problem],
    *,
    split: str,
    registry: ReasonerRegistry,
    token_counter: QwenTokenCounter,
    skipped: Counter[str],
    task_skipped: dict[str, Counter[str]],
    raw_traces: list[dict[str, Any]],
    reasoner_stats: dict[str, Counter[str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_prompt_hashes: set[str] = set()
    for index, problem in enumerate(problems):
        task_type = task_type_for_problem(problem)
        answer_row = make_answer_only_record(problem, index, token_counter)
        answer_row["id"] = f"stage3-{split}-wonderland-answer_only-{index:06d}-{problem.id}"
        _append_if_valid(rows, answer_row, skipped=skipped, seen_prompt_hashes=seen_prompt_hashes)

        reasoner_stats[task_type]["attempted"] += 1
        result = registry.run(problem)
        raw_traces.append(
            {
                "source_id": problem.id,
                "split": split,
                "task_type": task_type,
                "ok": result.ok,
                "error": result.error,
                "answer": result.answer,
                "gold_answer": problem.answer,
                "raw_trace": result.raw_trace,
            }
        )
        if not result.ok:
            reason = f"cot_{task_type}_{result.error or 'reasoner_failed'}"
            skipped[reason] += 1
            task_skipped[task_type][reason] += 1
            continue
        if result.answer.strip() != problem.answer.strip():
            skipped["cot_answer_mismatch"] += 1
            task_skipped[task_type]["cot_answer_mismatch"] += 1
            continue
        reasoner_stats[task_type]["ok"] += 1
        cot_row = make_cot_record(problem, result, index, token_counter)
        cot_row["id"] = f"stage3-{split}-wonderland-compressed_cot-{index:06d}-{problem.id}"
        before_len = len(rows)
        _append_if_valid(rows, cot_row, skipped=skipped, seen_prompt_hashes=seen_prompt_hashes)
        if len(rows) == before_len:
            task_skipped[task_type]["cot_validation_or_duplicate_rejected"] += 1
    return rows


def build_replay_rows_for_split(
    replay_rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    sample_type: str,
    limit: int,
    seed: int,
    token_counter: QwenTokenCounter,
    skipped: Counter[str],
) -> list[dict[str, Any]]:
    candidates = list(replay_rows)
    random.Random(seed).shuffle(candidates)
    if limit > 0:
        candidates = candidates[:limit]
    out: list[dict[str, Any]] = []
    seen_prompt_hashes: set[str] = set()
    for index, row in enumerate(candidates):
        record = make_replay_record(
            row,
            sample_type=sample_type,
            split=split,
            index=index,
            token_counter=token_counter,
        )
        _append_if_valid(out, record, skipped=skipped, seen_prompt_hashes=seen_prompt_hashes)
    return out


def _length_stats(values: Sequence[int]) -> dict[str, float]:
    if not values:
        return {"min": 0, "p50": 0, "p90": 0, "p95": 0, "max": 0, "mean": 0}
    sorted_values = sorted(values)

    def percentile(p: float) -> int:
        if len(sorted_values) == 1:
            return sorted_values[0]
        idx = int(round((len(sorted_values) - 1) * p))
        return sorted_values[idx]

    return {
        "min": min(sorted_values),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "max": max(sorted_values),
        "mean": round(statistics.fmean(sorted_values), 2),
    }


def _rate(ok: int, total: int) -> dict[str, Any]:
    return {"ok": ok, "total": total, "rate": round(ok / total, 6) if total else 1.0}


def build_report(
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    skipped: Counter[str],
    task_skipped: Mapping[str, Counter[str]],
    reasoner_stats: Mapping[str, Counter[str]],
    pool_ids: set[str],
    missing_split_ids: set[str],
    tokenizer: QwenTokenCounter,
) -> dict[str, Any]:
    all_rows = [row for rows in rows_by_split.values() for row in rows]
    sample_types = Counter(str(row["sample_type"]) for row in all_rows)
    task_types = Counter(str(row["task_type"]) for row in all_rows)
    token_lengths = [int(row["metadata"].get("token_length", 0)) for row in all_rows]
    parse_total = 0
    parse_ok = 0
    match_total = 0
    match_ok = 0
    think_total = 0
    think_ok = 0
    boxed_residue = 0
    leaked_source_ids: set[str] = set()
    for row in all_rows:
        assistant = row["messages"][1]["content"]
        if "\\boxed" in assistant:
            boxed_residue += 1
        if row["sample_type"] in {SAMPLE_ANSWER_ONLY, SAMPLE_COMPRESSED_COT}:
            parse_total += 1
            final = parse_final_answer(assistant)
            if final_answer_parse_ok(str(row["task_type"]), final):
                parse_ok += 1
            match_total += 1
            if final == row["metadata"].get("gold_answer"):
                match_ok += 1
            source_id = str(row["metadata"].get("source_id", ""))
            if source_id and source_id not in pool_ids:
                leaked_source_ids.add(source_id)
        if row["sample_type"] == SAMPLE_COMPRESSED_COT:
            think_total += 1
            if parse_thinking(assistant) is not None:
                think_ok += 1
    reasoner_rates = {
        task: _rate(counts.get("ok", 0), counts.get("attempted", 0))
        for task, counts in sorted(reasoner_stats.items())
    }
    ratios = {
        sample_type: round(count / len(all_rows), 6) if all_rows else 0.0
        for sample_type, count in sorted(sample_types.items())
    }
    task_audit: dict[str, Any] = {}
    for task in (TASK_GRAVITY, TASK_UNIT):
        task_rows = [row for row in all_rows if row["task_type"] == task]
        task_sample_types = Counter(str(row["sample_type"]) for row in task_rows)
        skipped_top = task_skipped.get(task, Counter()).most_common(10)
        counts = reasoner_stats.get(task, Counter())
        task_audit[task] = {
            "train_source_total": counts.get("attempted", 0),
            "answer_only": task_sample_types.get(SAMPLE_ANSWER_ONLY, 0),
            "compressed_cot": task_sample_types.get(SAMPLE_COMPRESSED_COT, 0),
            "reasoner_success_rate": _rate(counts.get("ok", 0), counts.get("attempted", 0)),
            "skipped_reason_top10": [{"reason": reason, "count": count} for reason, count in skipped_top],
        }
    return {
        "sample_total": len(all_rows),
        "splits": {split: len(rows) for split, rows in rows_by_split.items()},
        "task_type_counts": dict(sorted(task_types.items())),
        "sample_type_counts": dict(sorted(sample_types.items())),
        "sample_type_ratios": ratios,
        "task_audit": task_audit,
        "reasoner_success_rate_by_task_type": reasoner_rates,
        "skipped_reasons": dict(sorted((k, v) for k, v in skipped.items() if v)),
        "token_length_distribution": _length_stats(token_lengths),
        "parse_success": _rate(parse_ok, parse_total),
        "answer_match_rate": _rate(match_ok, match_total),
        "think_close_success": _rate(think_ok, think_total),
        "boxed_residue_count": boxed_residue,
        "source_split_leakage_check": {
            "ok": not leaked_source_ids,
            "required_split": "stage3_sft_pool",
            "stage3_sft_pool_count": len(pool_ids),
            "leaked_source_ids": sorted(leaked_source_ids),
            "missing_split_ids": sorted(missing_split_ids),
        },
        "tokenizer": {
            "kind": tokenizer.kind,
            "name": tokenizer.name,
            "max_token_length": MAX_TOKEN_LENGTH,
        },
        "constraints": {
            "read_wonderland_validation_or_test": False,
            "used_only_stage3_sft_pool": not leaked_source_ids,
            "trained_model": False,
            "stage1_5_or_stage2_adapter_overwritten": False,
            "raw_trace_used_in_training_samples": False,
        },
    }


def build_audit_md(report: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage 3 Wonderland Cold-Start Audit",
            "",
            f"- sample_total: {report['sample_total']}",
            f"- splits: {json.dumps(report['splits'], ensure_ascii=False, sort_keys=True)}",
            f"- task_type_counts: {json.dumps(report['task_type_counts'], ensure_ascii=False, sort_keys=True)}",
            f"- sample_type_counts: {json.dumps(report['sample_type_counts'], ensure_ascii=False, sort_keys=True)}",
            f"- sample_type_ratios: {json.dumps(report['sample_type_ratios'], ensure_ascii=False, sort_keys=True)}",
            f"- task_audit: {json.dumps(report['task_audit'], ensure_ascii=False, sort_keys=True)}",
            f"- reasoner_success_rate_by_task_type: {json.dumps(report['reasoner_success_rate_by_task_type'], ensure_ascii=False, sort_keys=True)}",
            f"- skipped_reasons: {json.dumps(report['skipped_reasons'], ensure_ascii=False, sort_keys=True)}",
            f"- token_length_distribution: {json.dumps(report['token_length_distribution'], ensure_ascii=False, sort_keys=True)}",
            f"- parse_success: {json.dumps(report['parse_success'], ensure_ascii=False, sort_keys=True)}",
            f"- answer_match_rate: {json.dumps(report['answer_match_rate'], ensure_ascii=False, sort_keys=True)}",
            f"- think_close_success: {json.dumps(report['think_close_success'], ensure_ascii=False, sort_keys=True)}",
            f"- boxed_residue_count: {report['boxed_residue_count']}",
            f"- source_split_leakage_check: {json.dumps(report['source_split_leakage_check'], ensure_ascii=False, sort_keys=True)}",
            "",
            "Confirmed: no training was started; no adapter path was written; Wonderland validation/test files were not read.",
            "",
        ]
    )


def build_manual_review_md(
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    raw_traces: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
) -> str:
    lines = [
        "# Stage 3 Manual Review",
        "",
        "## Summary",
        "",
        f"- sample_total: {report['sample_total']}",
        f"- raw_trace_debug_rows: {len(raw_traces)}",
        "",
        "## Sample Rows",
        "",
    ]

    def append_row(row: Mapping[str, Any], split: str, number: int) -> None:
        user, assistant = get_user_and_assistant(row["messages"])
        lines.extend(
            [
                f"### {number}. {split} / {row['sample_type']} / {row['task_type']}",
                "",
                f"- id: `{row['id']}`",
                f"- token_length: {row['metadata'].get('token_length')}",
                "",
                "User:",
                "```text",
                user[:1200],
                "```",
                "",
                "Assistant:",
                "```text",
                assistant[:1200],
                "```",
                "",
            ]
        )

    shown = 0
    for split, rows in rows_by_split.items():
        for row in rows[:5]:
            shown += 1
            append_row(row, split, shown)
    if shown == 0:
        lines.append("No rows generated.")

    lines.extend(["", "## Gravity Compressed-CoT Samples", ""])
    gravity_rows: list[tuple[str, Mapping[str, Any]]] = []
    unit_rows: list[tuple[str, Mapping[str, Any]]] = []
    for split, rows in rows_by_split.items():
        for row in rows:
            if row["sample_type"] != SAMPLE_COMPRESSED_COT:
                continue
            if row["task_type"] == TASK_GRAVITY and len(gravity_rows) < 10:
                gravity_rows.append((split, row))
            if row["task_type"] == TASK_UNIT and len(unit_rows) < 10:
                unit_rows.append((split, row))
    if gravity_rows:
        for idx, (split, row) in enumerate(gravity_rows, start=1):
            append_row(row, split, idx)
    else:
        lines.append("No gravity compressed-CoT rows generated.")

    lines.extend(["", "## Unit Conversion Compressed-CoT Samples", ""])
    if unit_rows:
        for idx, (split, row) in enumerate(unit_rows, start=1):
            append_row(row, split, idx)
    else:
        lines.append("No unit_conversion compressed-CoT rows generated.")
    return "\n".join(lines) + "\n"


def validate_all_splits(rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    ids: set[str] = set()
    for split, rows in rows_by_split.items():
        split_hashes: set[str] = set()
        for row in rows:
            rid = str(row["id"])
            if rid in ids:
                raise ValueError(f"duplicate id: {rid}")
            ids.add(rid)
            validate_stage3_record(row)
            h = str(row["metadata"]["prompt_hash"])
            if h in split_hashes:
                raise ValueError(f"duplicate prompt hash in {split}: {h[:12]}")
            split_hashes.add(h)


def generate_stage3_dataset(
    *,
    input_csv: Path = DEFAULT_INPUT_CSV,
    split_path: Path = DEFAULT_SPLIT_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    seed: int = 42,
    dev_ratio: float = 0.05,
    max_stage3_prompts: int = 0,
    stage1_5_replay_path: Path = DEFAULT_STAGE1_5_REPLAY,
    stage2_replay_path: Path = DEFAULT_STAGE2_REPLAY,
    replay_train_limit: int = 200,
    replay_dev_limit: int = 40,
    tokenizer: Any | None = None,
    tokenizer_model: str | None = None,
    tokenizer_cache_dir: Path | None = None,
    dry_run: bool = False,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    token_counter = QwenTokenCounter(
        tokenizer,
        model_id=tokenizer_model,
        cache_dir=tokenizer_cache_dir,
    )
    skipped: Counter[str] = Counter()
    task_skipped: dict[str, Counter[str]] = defaultdict(Counter)
    reasoner_stats: dict[str, Counter[str]] = defaultdict(Counter)
    raw_traces: list[dict[str, Any]] = []
    registry = build_reasoner_registry()

    problems, pool_ids, missing_split_ids = select_wonderland_problems(
        input_csv=input_csv,
        split_path=split_path,
        max_stage3_prompts=max_stage3_prompts,
        seed=seed,
        skipped=skipped,
    )
    train_problems, dev_problems = _partition_train_dev(problems, dev_ratio)

    stage1_5_rows = load_replay_rows(
        stage1_5_replay_path,
        sample_type=SAMPLE_STAGE1_5_REPLAY,
        skipped=skipped,
    )
    stage2_rows = load_replay_rows(
        stage2_replay_path,
        sample_type=SAMPLE_STAGE2_REPLAY,
        skipped=skipped,
    )

    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "dev": []}
    rows_by_split["train"].extend(
        build_wonderland_rows_for_split(
            train_problems,
            split="train",
            registry=registry,
            token_counter=token_counter,
            skipped=skipped,
            task_skipped=task_skipped,
            raw_traces=raw_traces,
            reasoner_stats=reasoner_stats,
        )
    )
    rows_by_split["dev"].extend(
        build_wonderland_rows_for_split(
            dev_problems,
            split="dev",
            registry=registry,
            token_counter=token_counter,
            skipped=skipped,
            task_skipped=task_skipped,
            raw_traces=raw_traces,
            reasoner_stats=reasoner_stats,
        )
    )

    for split, limit in (("train", replay_train_limit), ("dev", replay_dev_limit)):
        if limit <= 0:
            continue
        rows_by_split[split].extend(
            build_replay_rows_for_split(
                stage1_5_rows,
                split=split,
                sample_type=SAMPLE_STAGE1_5_REPLAY,
                limit=limit,
                seed=seed + (11 if split == "train" else 12),
                token_counter=token_counter,
                skipped=skipped,
            )
        )
        rows_by_split[split].extend(
            build_replay_rows_for_split(
                stage2_rows,
                split=split,
                sample_type=SAMPLE_STAGE2_REPLAY,
                limit=limit,
                seed=seed + (21 if split == "train" else 22),
                token_counter=token_counter,
                skipped=skipped,
            )
        )

    # Deduplicate by prompt_hash per split (stage1_5 and stage2 replay share prompts)
    for split_name in rows_by_split:
        seen: set[str] = set()
        deduped: list[dict[str, Any]] = []
        for r in rows_by_split[split_name]:
            h = str(r["metadata"]["prompt_hash"])
            if h in seen:
                skipped["duplicate_prompt_hash_across_replay"] += 1
                continue
            seen.add(h)
            deduped.append(r)
        rows_by_split[split_name] = deduped

    validate_all_splits(rows_by_split)
    report = build_report(
        rows_by_split,
        skipped=skipped,
        task_skipped=task_skipped,
        reasoner_stats=reasoner_stats,
        pool_ids=pool_ids,
        missing_split_ids=missing_split_ids,
        tokenizer=token_counter,
    )
    if not dry_run:
        write_jsonl_atomic(output_dir / "train.jsonl", rows_by_split["train"])
        write_jsonl_atomic(output_dir / "dev.jsonl", rows_by_split["dev"])
        write_json(output_dir / "report.json", report)
        write_text_atomic(output_dir / "audit.md", build_audit_md(report))
        write_text_atomic(output_dir / "manual_review.md", build_manual_review_md(rows_by_split, raw_traces, report))
        write_jsonl_atomic(output_dir / "debug" / "raw_traces.jsonl", raw_traces)
    return rows_by_split, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Stage 3 Wonderland cold-start data.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-ratio", type=float, default=0.05)
    parser.add_argument(
        "--max-stage3-prompts",
        type=int,
        default=0,
        help="0 means use all IDs in stage3_sft_pool.",
    )
    parser.add_argument("--stage1-5-replay-path", type=Path, default=DEFAULT_STAGE1_5_REPLAY)
    parser.add_argument("--stage2-replay-path", type=Path, default=DEFAULT_STAGE2_REPLAY)
    parser.add_argument("--replay-train-limit", type=int, default=200)
    parser.add_argument("--replay-dev-limit", type=int, default=40)
    parser.add_argument("--tokenizer-model", type=str, default=None)
    parser.add_argument("--tokenizer-cache-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_stage3_prompts < 0:
        raise ValueError("--max-stage3-prompts must be >= 0")
    if not args.input_csv.exists():
        raise FileNotFoundError(f"input CSV not found: {args.input_csv}")
    log(
        f"seed={args.seed} input_csv={args.input_csv} split_path={args.split_path} "
        f"output_dir={args.output_dir} max_stage3_prompts={args.max_stage3_prompts} "
        f"dry_run={args.dry_run}"
    )
    rows_by_split, report = generate_stage3_dataset(
        input_csv=args.input_csv,
        split_path=args.split_path,
        output_dir=args.output_dir,
        seed=args.seed,
        dev_ratio=args.dev_ratio,
        max_stage3_prompts=args.max_stage3_prompts,
        stage1_5_replay_path=args.stage1_5_replay_path,
        stage2_replay_path=args.stage2_replay_path,
        replay_train_limit=args.replay_train_limit,
        replay_dev_limit=args.replay_dev_limit,
        tokenizer_model=args.tokenizer_model,
        tokenizer_cache_dir=args.tokenizer_cache_dir,
        dry_run=args.dry_run,
    )
    log("=== Stage 3 数据报告 ===")
    log(f"sample_total={report['sample_total']}")
    log(f"splits={report['splits']}")
    log(f"task_type_counts={report['task_type_counts']}")
    log(f"sample_type_counts={report['sample_type_counts']}")
    log(f"reasoner_success_rate_by_task_type={report['reasoner_success_rate_by_task_type']}")
    log(f"skipped_reasons={report['skipped_reasons']}")
    log(f"token_length_distribution={report['token_length_distribution']}")
    log(f"source_split_leakage_check={report['source_split_leakage_check']}")
    log("确认：未读取 Wonderland validation/test；未启动训练；未覆盖 Stage 1.5 或 Stage 2 adapter。")
    if args.dry_run:
        log("--dry-run 模式：不写入文件。")
    else:
        log(f"wrote {args.output_dir / 'train.jsonl'} ({len(rows_by_split['train'])} rows)")
        log(f"wrote {args.output_dir / 'dev.jsonl'} ({len(rows_by_split['dev'])} rows)")
        log(f"wrote {args.output_dir / 'report.json'}")
        log(f"wrote {args.output_dir / 'audit.md'}")
        log(f"wrote {args.output_dir / 'manual_review.md'}")
        log(f"wrote {args.output_dir / 'debug' / 'raw_traces.jsonl'}")


if __name__ == "__main__":
    main()
