#!/usr/bin/env python3
"""Generate Stage 3 Wonderland cold-start SFT data.

Stage 3 only consumes Wonderland train prompts. It does not read validation/test
files and it does not start training or touch adapter directories.

Output:
    data/instruction/stage3_wonderland_cold_start/train.jsonl
    data/instruction/stage3_wonderland_cold_start/report.json

Rows use messages format:
    [{"role":"user","content":prompt},{"role":"assistant","content":completion}]
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
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
TASK_TYPE = "bit_manipulation"
MAX_TOKEN_LENGTH = 1024
DEFAULT_INPUT_CSV = Path("data/raw/train.csv")
DEFAULT_OUTPUT_DIR = Path("data/instruction/stage3_wonderland_cold_start")
DEFAULT_MAX_PROMPTS = 512

BINARY_ANSWER_RE = re.compile(r"[01]{8}")
BINARY_EXAMPLE_RE = re.compile(r"([01]{8}) -> ([01]{8})")
BINARY_QUERY_RE = re.compile(r"Now, determine the output for: ([01]{8})")
BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")

BIT_PREFIX = (
    "In Alice's Wonderland, a secret bit manipulation rule transforms "
    "8-bit binary numbers. The transformation involves operations like "
    "bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority or "
    "choice functions."
)


@dataclass(frozen=True)
class Example:
    input_value: str
    output_value: str


@dataclass(frozen=True)
class BitProblem:
    id: str
    category: str
    examples: list[Example]
    question: str
    answer: str
    prompt: str


@dataclass(frozen=True)
class ReasonerResult:
    answer: str
    compressed_trace: str
    ok: bool
    error: str
    confidence: str
    rule_vector: tuple[str, ...] = ()


def log(message: str) -> None:
    timestamp = datetime.now().astimezone().strftime("%H:%M:%S")
    print(f"[{timestamp}] [stage3] {message}", file=sys.stderr, flush=True)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def prompt_hash(text: str) -> str:
    normalized = normalize_whitespace(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def make_messages(user: str, assistant: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(tmp, path)


def count_tokens(text: str) -> int:
    if not text:
        return 0
    # Conservative heuristic used only as a local data-quality gate.
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    non_ascii = len(text) - ascii_chars
    return max(1, int(ascii_chars / 3.2) + non_ascii)


def parse_bit_problem(record_id: str, prompt: str, answer: str) -> BitProblem:
    lines = prompt.splitlines()
    if not lines or lines[0] != BIT_PREFIX:
        raise ValueError("not_bit_manipulation")
    examples = [Example(a, b) for a, b in BINARY_EXAMPLE_RE.findall(prompt)]
    if not 7 <= len(examples) <= 10:
        raise ValueError(f"unexpected_example_count:{len(examples)}")
    query_match = BINARY_QUERY_RE.search(prompt)
    if query_match is None:
        raise ValueError("missing_bit_query")
    if BINARY_ANSWER_RE.fullmatch(answer) is None:
        raise ValueError("invalid_gold_answer")
    return BitProblem(
        id=record_id,
        category=TASK_TYPE,
        examples=examples,
        question=query_match.group(1),
        answer=answer,
        prompt=prompt,
    )


def _load_nemotron_bit_reasoner() -> Any:
    script_path = Path(__file__).resolve()
    nemotron_root = script_path.parents[1].parent / "nemotron"
    if str(nemotron_root) not in sys.path:
        sys.path.insert(0, str(nemotron_root))
    try:
        from reasoners.bit_manipulation import reasoning_bit_manipulation
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"failed_to_import_bit_reasoner:{exc}") from exc
    return reasoning_bit_manipulation


def _extract_reasoner_answer(reasoning_text: str) -> str:
    matches = [m.strip() for m in BOXED_RE.findall(reasoning_text)]
    matches = [m for m in matches if m and m != "–"]
    return matches[-1] if matches else ""


def _extract_selected_rules(reasoning_text: str) -> tuple[str, ...]:
    lines = reasoning_text.splitlines()
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


def _build_compressed_trace(problem: BitProblem, rules: Sequence[str], answer: str) -> str:
    rule_text = ", ".join(f"b{i}:{rule}" for i, rule in enumerate(rules))
    if not rule_text:
        rule_text = "rule vector unavailable"
    return "\n".join(
        [
            "Task type: bit_manipulation over 8-bit binary strings.",
            f"Selected 8-bit rule: {rule_text}.",
            f"Apply the selected rule bit-by-bit to query {problem.question}.",
            f"This produces {answer}.",
        ]
    )


def reason_bit_manipulation(problem: BitProblem) -> ReasonerResult:
    try:
        reasoner = _load_nemotron_bit_reasoner()
        reasoning_text = reasoner(problem)
    except Exception as exc:  # noqa: BLE001
        return ReasonerResult("", "", False, f"reasoner_error:{exc}", "low")
    if not reasoning_text:
        return ReasonerResult("", "", False, "reasoner_returned_none", "low")
    answer = _extract_reasoner_answer(reasoning_text)
    if BINARY_ANSWER_RE.fullmatch(answer) is None:
        return ReasonerResult(answer, "", False, "invalid_reasoner_answer", "low")
    rules = _extract_selected_rules(reasoning_text)
    trace = _build_compressed_trace(problem, rules, answer)
    confidence = "high" if len(rules) == 8 else "medium"
    return ReasonerResult(answer, trace, True, "", confidence, rules)


def build_answer_only_prompt(problem: BitProblem) -> str:
    return problem.prompt.rstrip() + "\n\nReturn only the final 8-bit answer."


def build_cot_prompt(problem: BitProblem) -> str:
    return (
        problem.prompt.rstrip()
        + f"\n\nThink briefly in {THINK_OPEN}...{THINK_CLOSE}, then output only the final 8-bit answer."
    )


def build_thinking_completion(compressed_trace: str, final_answer: str) -> str:
    return f"{THINK_OPEN}\n{compressed_trace.strip()}\n{THINK_CLOSE}\n\n{final_answer.strip()}"


def parse_final_answer(completion: str) -> str:
    if THINK_OPEN in completion or THINK_CLOSE in completion:
        parsed = parse_thinking(completion)
        if parsed is None:
            return ""
        return parsed[1]
    return completion.strip()


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


def _length_stats(values: Sequence[int]) -> dict[str, float]:
    if not values:
        return {"min": 0, "p50": 0, "p95": 0, "max": 0, "mean": 0}
    sorted_values = sorted(values)

    def percentile(p: float) -> int:
        if len(sorted_values) == 1:
            return sorted_values[0]
        idx = int(round((len(sorted_values) - 1) * p))
        return sorted_values[idx]

    return {
        "min": min(sorted_values),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": max(sorted_values),
        "mean": round(statistics.fmean(sorted_values), 2),
    }


def _base_record(
    *,
    record_id: str,
    problem: BitProblem,
    sample_type: str,
    user: str,
    assistant: str,
    reasoner_result: ReasonerResult | None,
) -> dict[str, Any]:
    token_length = count_tokens(user) + count_tokens(assistant)
    return {
        "id": record_id,
        "task_type": TASK_TYPE,
        "sample_type": sample_type,
        "messages": make_messages(user, assistant),
        "metadata": {
            "source_id": problem.id,
            "source_prompt_hash": prompt_hash(problem.prompt),
            "prompt_hash": prompt_hash(user),
            "gold_answer": problem.answer,
            "final_answer": parse_final_answer(assistant),
            "token_length": token_length,
            "reasoner": {
                "ok": reasoner_result.ok if reasoner_result else False,
                "error": reasoner_result.error if reasoner_result else "",
                "confidence": reasoner_result.confidence if reasoner_result else "",
                "answer": reasoner_result.answer if reasoner_result else "",
            },
        },
    }


def make_answer_only_record(problem: BitProblem, index: int) -> dict[str, Any]:
    return _base_record(
        record_id=f"stage3-answer_only-{index:05d}-{problem.id}",
        problem=problem,
        sample_type="answer_only",
        user=build_answer_only_prompt(problem),
        assistant=problem.answer,
        reasoner_result=None,
    )


def make_cot_record(problem: BitProblem, result: ReasonerResult, index: int) -> dict[str, Any]:
    return _base_record(
        record_id=f"stage3-compressed_cot-{index:05d}-{problem.id}",
        problem=problem,
        sample_type="compressed_cot",
        user=build_cot_prompt(problem),
        assistant=build_thinking_completion(result.compressed_trace, result.answer),
        reasoner_result=result,
    )


def validate_stage3_record(row: Mapping[str, Any]) -> None:
    for field in ("id", "task_type", "sample_type", "messages", "metadata"):
        if field not in row:
            raise ValueError(f"missing field: {field}")
    if row["task_type"] != TASK_TYPE:
        raise ValueError(f"unsupported task_type: {row['task_type']!r}")
    if row["sample_type"] not in {"answer_only", "compressed_cot"}:
        raise ValueError(f"unsupported sample_type: {row['sample_type']!r}")
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
    if not isinstance(user, str) or not user.strip():
        raise ValueError("empty user content")
    if not isinstance(assistant, str) or not assistant.strip():
        raise ValueError("empty assistant content")
    if "<|im_end|>" in assistant:
        raise ValueError("assistant must not contain <|im_end|>")
    if row["sample_type"] == "compressed_cot":
        parsed = parse_thinking(assistant)
        if parsed is None:
            raise ValueError("compressed CoT sample missing closed think tags")
        _, tail = parsed
        if THINK_OPEN in tail or THINK_CLOSE in tail:
            raise ValueError("think tags after close are not allowed")
        if "\n" in tail:
            raise ValueError("after </think> there must be only final answer")
    final = parse_final_answer(assistant)
    if BINARY_ANSWER_RE.fullmatch(final) is None:
        raise ValueError(f"final answer parse failed: {final!r}")
    if final != meta.get("gold_answer"):
        raise ValueError(f"answer != gold: {final!r} != {meta.get('gold_answer')!r}")
    if final != meta.get("final_answer"):
        raise ValueError("metadata final_answer mismatch")
    if row["sample_type"] == "answer_only":
        if THINK_OPEN in assistant or THINK_CLOSE in assistant:
            raise ValueError("answer-only sample must not contain think tags")
    else:
        assert parsed is not None
        reasoning, tail = parsed
        if tail != final:
            raise ValueError("after </think> must contain only final answer")
        if "\\boxed" in reasoning:
            raise ValueError("compressed trace must not reuse boxed long reasoning")
    if int(meta.get("token_length", 0)) > MAX_TOKEN_LENGTH:
        raise ValueError(f"token length exceeds {MAX_TOKEN_LENGTH}")


def _read_train_csv(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["id", "prompt", "answer"]:
            raise ValueError(f"unexpected CSV fields: {reader.fieldnames}")
        for row in reader:
            rows.append({"id": row["id"], "prompt": row["prompt"], "answer": row["answer"]})
    return rows


def _select_problems(
    rows: Sequence[Mapping[str, str]],
    *,
    max_prompts: int,
    seed: int,
    skipped: Counter[str],
) -> list[BitProblem]:
    parsed: list[BitProblem] = []
    seen_source_hashes: set[str] = set()
    for row in rows:
        try:
            problem = parse_bit_problem(row["id"], row["prompt"], row["answer"])
        except ValueError as exc:
            reason = str(exc)
            if reason == "not_bit_manipulation":
                skipped["unsupported_task_type"] += 1
            else:
                skipped[reason] += 1
            continue
        source_hash = prompt_hash(problem.prompt)
        if source_hash in seen_source_hashes:
            skipped["duplicate_source_prompt_hash"] += 1
            continue
        seen_source_hashes.add(source_hash)
        parsed.append(problem)
    rng = random.Random(seed)
    rng.shuffle(parsed)
    selected = parsed[:max_prompts]
    skipped["reserved_for_rl"] += max(0, len(parsed) - len(selected))
    return selected


def _validate_all(rows: Sequence[Mapping[str, Any]]) -> None:
    prompt_hashes: set[str] = set()
    ids: set[str] = set()
    for row in rows:
        rid = str(row["id"])
        if rid in ids:
            raise ValueError(f"duplicate id: {rid}")
        ids.add(rid)
        validate_stage3_record(row)
        h = str(row["metadata"]["prompt_hash"])
        if h in prompt_hashes:
            raise ValueError(f"duplicate prompt hash: {h[:12]}")
        prompt_hashes.add(h)


def build_report(rows: Sequence[Mapping[str, Any]], skipped: Counter[str], selected_prompts: int) -> dict[str, Any]:
    sample_types = Counter(str(row["sample_type"]) for row in rows)
    task_types = Counter(str(row["task_type"]) for row in rows)
    token_lengths = [int(row["metadata"]["token_length"]) for row in rows]
    cot_conf = Counter(
        str(row["metadata"]["reasoner"].get("confidence", ""))
        for row in rows
        if row["sample_type"] == "compressed_cot"
    )
    return {
        "sample_total": len(rows),
        "selected_source_prompts": selected_prompts,
        "task_type_counts": dict(sorted(task_types.items())),
        "sample_type_counts": dict(sorted(sample_types.items())),
        "skipped_reasons": dict(sorted((k, v) for k, v in skipped.items() if v)),
        "token_length_distribution": _length_stats(token_lengths),
        "cot_reasoner_confidence": dict(sorted(cot_conf.items())),
        "constraints": {
            "used_wonderland_train_only": True,
            "read_wonderland_validation_or_test": False,
            "max_token_length": MAX_TOKEN_LENGTH,
            "stage1_5_or_stage2_adapter_overwritten": False,
        },
    }


def generate_stage3_dataset(
    *,
    input_csv: Path,
    output_dir: Path,
    max_prompts: int = DEFAULT_MAX_PROMPTS,
    seed: int = 42,
    dry_run: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    skipped: Counter[str] = Counter()
    raw_rows = _read_train_csv(input_csv)
    problems = _select_problems(raw_rows, max_prompts=max_prompts, seed=seed, skipped=skipped)

    rows: list[dict[str, Any]] = []
    answer_idx = 0
    cot_idx = 0
    for problem in problems:
        answer_row = make_answer_only_record(problem, answer_idx)
        answer_idx += 1
        if answer_row["metadata"]["token_length"] <= MAX_TOKEN_LENGTH:
            rows.append(answer_row)
        else:
            skipped["answer_only_token_too_long"] += 1

        result = reason_bit_manipulation(problem)
        if not result.ok:
            skipped[f"cot_{result.error or 'reasoner_failed'}"] += 1
            continue
        if result.answer != problem.answer:
            skipped["cot_answer_mismatch"] += 1
            continue
        cot_row = make_cot_record(problem, result, cot_idx)
        cot_idx += 1
        if cot_row["metadata"]["token_length"] > MAX_TOKEN_LENGTH:
            skipped["cot_token_too_long"] += 1
            continue
        rows.append(cot_row)

    _validate_all(rows)
    report = build_report(rows, skipped, len(problems))
    if not dry_run:
        write_jsonl_atomic(output_dir / "train.jsonl", rows)
        write_json(output_dir / "report.json", report)
    return rows, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Stage 3 Wonderland cold-start data.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-prompts", type=int, default=DEFAULT_MAX_PROMPTS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_prompts <= 0:
        raise ValueError("--max-prompts must be positive")
    if not args.input_csv.exists():
        raise ValueError(f"input CSV not found: {args.input_csv}")
    log(
        f"seed={args.seed} input_csv={args.input_csv} output_dir={args.output_dir} "
        f"max_prompts={args.max_prompts} dry_run={args.dry_run}"
    )
    rows, report = generate_stage3_dataset(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        max_prompts=args.max_prompts,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    log("=== Stage 3 数据报告 ===")
    log(f"sample_total={report['sample_total']}")
    log(f"selected_source_prompts={report['selected_source_prompts']}")
    log(f"task_type_counts={report['task_type_counts']}")
    log(f"sample_type_counts={report['sample_type_counts']}")
    log(f"skipped_reasons={report['skipped_reasons']}")
    log(f"token_length_distribution={report['token_length_distribution']}")
    log("确认：未读取 Wonderland validation/test；未启动训练；未覆盖 Stage 1.5 或 Stage 2 adapter。")
    if args.dry_run:
        log("--dry-run 模式：不写入文件。")
    else:
        log(f"wrote {args.output_dir / 'train.jsonl'} ({len(rows)} rows)")
        log(f"wrote {args.output_dir / 'report.json'}")


if __name__ == "__main__":
    main()
