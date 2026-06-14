#!/usr/bin/env python3
"""Analyze tokenizer lengths without loading model weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence


SPLITS = ("train", "validation", "test")
REQUIRED_FIELDS = ("id", "task_type", "prompt", "answer")
LENGTH_FIELDS = (
    "prompt_tokens",
    "answer_tokens",
    "full_sequence_tokens",
)
THRESHOLDS = (512, 768, 1024, 1536, 2048, 4096)
MAX_LENGTH_CANDIDATES = THRESHOLDS
TEMPLATE_NAME = "plain_answer_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path, split: str) -> list[dict[str, str]]:
    """Read one split and attach its split name without mutating input data."""
    records: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number} contains invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"{path}:{line_number} must contain a JSON object"
                )
            missing = [
                field for field in REQUIRED_FIELDS if field not in value
            ]
            if missing:
                raise ValueError(
                    f"{path}:{line_number} missing required fields: "
                    f"{', '.join(missing)}"
                )
            invalid_types = [
                field
                for field in REQUIRED_FIELDS
                if not isinstance(value[field], str)
            ]
            if invalid_types:
                raise ValueError(
                    f"{path}:{line_number} fields must be strings: "
                    f"{', '.join(invalid_types)}"
                )
            records.append(
                {
                    "id": value["id"],
                    "task_type": value["task_type"],
                    "prompt": value["prompt"],
                    "answer": value["answer"],
                    "split": split,
                }
            )
    if not records:
        raise ValueError(f"{path} contains no records")
    return records


def load_datasets(
    data_dir: Path,
) -> tuple[list[dict[str, str]], dict[str, dict[str, Any]]]:
    records: list[dict[str, str]] = []
    input_files: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        path = data_dir / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"missing input file: {path}")
        split_records = read_jsonl(path, split)
        records.extend(split_records)
        input_files[split] = {
            "path": str(path),
            "count": len(split_records),
            "sha256_before": sha256_file(path),
        }
    return records, input_files


def format_training_text(
    prompt: str,
    answer: str,
    eos_token: str,
) -> tuple[str, str, str]:
    """
    Return the prompt, answer, and full text for the candidate SFT format.

    The source prompt and answer are used verbatim.
    """
    if not eos_token:
        raise ValueError("tokenizer EOS token is required")
    prompt_text = f"{prompt}\n\nAnswer:\n"
    answer_text = f"{answer}{eos_token}"
    return prompt_text, answer_text, prompt_text + answer_text


def _token_count(tokenizer: Any, text: str) -> int:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    return len(encoded["input_ids"])


def analyze_records(
    records: Sequence[Mapping[str, str]], tokenizer: Any
) -> list[dict[str, Any]]:
    eos_token = tokenizer.eos_token
    if not eos_token:
        raise ValueError("tokenizer has no EOS token")

    analyzed: list[dict[str, Any]] = []
    for record in records:
        prompt_text, answer_text, full_text = format_training_text(
            record["prompt"], record["answer"], eos_token
        )
        analyzed.append(
            {
                "id": record["id"],
                "split": record["split"],
                "task_type": record["task_type"],
                "prompt_tokens": _token_count(tokenizer, prompt_text),
                "answer_tokens": _token_count(tokenizer, answer_text),
                # Tokenize the joined text independently because token
                # boundaries can change where prompt and answer meet.
                "full_sequence_tokens": _token_count(tokenizer, full_text),
            }
        )
    if not analyzed:
        raise ValueError("cannot analyze an empty dataset")
    return analyzed


def percentile(values: Sequence[int | float], quantile: float) -> float | int:
    """Return a linearly interpolated percentile, matching NumPy's default."""
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between 0 and 1")
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    fraction = position - lower_index
    return (
        sorted_values[lower_index] * (1 - fraction)
        + sorted_values[upper_index] * fraction
    )


def _rounded(value: float | int) -> float | int:
    if isinstance(value, int):
        return value
    return round(value, 2)


def summarize_values(values: Sequence[int]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize empty values")
    return {
        "count": len(values),
        "min": min(values),
        "mean": round(mean(values), 2),
        "median": _rounded(median(values)),
        "p90": _rounded(percentile(values, 0.90)),
        "p95": _rounded(percentile(values, 0.95)),
        "p99": _rounded(percentile(values, 0.99)),
        "max": max(values),
    }


def summarize_group(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float | int]]:
    if not records:
        raise ValueError("cannot summarize an empty record group")
    return {
        field: summarize_values([int(record[field]) for record in records])
        for field in LENGTH_FIELDS
    }


def _group_by(
    records: Sequence[Mapping[str, Any]], key: str
) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record[key])].append(record)
    return dict(sorted(groups.items()))


def build_statistics(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_split = _group_by(records, "split")
    by_task = _group_by(records, "task_type")
    return {
        "overall": summarize_group(records),
        "by_split": {
            split: summarize_group(group)
            for split, group in by_split.items()
        },
        "by_task_type": {
            task: summarize_group(group)
            for task, group in by_task.items()
        },
        "by_split_task_type": {
            split: {
                task: summarize_group(task_group)
                for task, task_group in _group_by(
                    split_group, "task_type"
                ).items()
            }
            for split, split_group in by_split.items()
        },
    }


def _overflow_summary(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    count = len(records)
    return {
        str(threshold): {
            "count": overflow_count,
            "ratio": round(overflow_count / count, 8),
        }
        for threshold in THRESHOLDS
        for overflow_count in [
            sum(
                int(record["full_sequence_tokens"]) > threshold
                for record in records
            )
        ]
    }


def build_overflow_statistics(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_split = _group_by(records, "split")
    by_task = _group_by(records, "task_type")
    return {
        "thresholds": list(THRESHOLDS),
        "comparison": "full_sequence_tokens > threshold",
        "overall": _overflow_summary(records),
        "by_split": {
            split: _overflow_summary(group)
            for split, group in by_split.items()
        },
        "by_task_type": {
            task: _overflow_summary(group)
            for task, group in by_task.items()
        },
        "by_split_task_type": {
            split: {
                task: _overflow_summary(task_group)
                for task, task_group in _group_by(
                    split_group, "task_type"
                ).items()
            }
            for split, split_group in by_split.items()
        },
    }


def recommend_max_length(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    development_records = [
        record
        for record in records
        if record["split"] in {"train", "validation"}
    ]
    if not development_records:
        raise ValueError("train and validation records are required")

    candidate_results = []
    for candidate in MAX_LENGTH_CANDIDATES:
        overflow_count = sum(
            int(record["full_sequence_tokens"]) > candidate
            for record in development_records
        )
        candidate_results.append(
            {
                "max_length": candidate,
                "overflow_count": overflow_count,
                "overflow_ratio": round(
                    overflow_count / len(development_records), 8
                ),
            }
        )

    complete_candidate = next(
        (
            result
            for result in candidate_results
            if result["overflow_count"] == 0
        ),
        None,
    )
    selected = complete_candidate or candidate_results[-1]
    if complete_candidate:
        reason = (
            f"{selected['max_length']} is the smallest allowed candidate "
            "that fully covers train and validation. Choosing the smallest "
            "complete candidate limits sequence-length memory and compute "
            "cost on a 24GB RTX 4090."
        )
    else:
        reason = (
            "No allowed candidate fully covers train and validation. "
            "4096 minimizes overflow among the allowed values; remaining "
            "samples require an explicit truncation or filtering decision."
        )

    previous = next(
        (
            result
            for result in reversed(candidate_results)
            if result["max_length"] < selected["max_length"]
        ),
        None,
    )
    long_tail_note = None
    if previous and previous["overflow_count"]:
        long_tail_note = (
            f"The next smaller candidate ({previous['max_length']}) would "
            f"overflow {previous['overflow_count']} development samples "
            f"({previous['overflow_ratio']:.6%})."
        )

    return {
        "based_on_splits": ["train", "validation"],
        "development_count": len(development_records),
        "candidates": candidate_results,
        "recommended_max_length": selected["max_length"],
        "overflow_count": selected["overflow_count"],
        "overflow_ratio": selected["overflow_ratio"],
        "reason": reason,
        "long_tail_note": long_tail_note,
    }


def collect_tokenizer_metadata(
    tokenizer: Any, model_id: str
) -> dict[str, Any]:
    init_kwargs = getattr(tokenizer, "init_kwargs", {}) or {}
    revision = (
        init_kwargs.get("_commit_hash")
        or getattr(tokenizer, "_commit_hash", None)
        or init_kwargs.get("revision")
        or getattr(tokenizer, "_analysis_revision", None)
    )
    try:
        from transformers import __version__ as transformers_version
    except ImportError:
        transformers_version = None
    return {
        "model_id": model_id,
        "tokenizer_class": tokenizer.__class__.__name__,
        "vocabulary_size": getattr(tokenizer, "vocab_size", None),
        "tokenizer_length": len(tokenizer),
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token": tokenizer.bos_token,
        "bos_token_id": tokenizer.bos_token_id,
        "revision": revision,
        "transformers_version": transformers_version,
    }


def build_report(
    records: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    model_id: str,
    input_files: Mapping[str, Mapping[str, Any]],
    execution_time: str | None = None,
) -> dict[str, Any]:
    longest = sorted(
        records,
        key=lambda record: (
            -int(record["full_sequence_tokens"]),
            str(record["id"]),
        ),
    )[:20]
    longest_samples = [
        {
            "id": record["id"],
            "split": record["split"],
            "task_type": record["task_type"],
            "prompt_tokens": record["prompt_tokens"],
            "answer_tokens": record["answer_tokens"],
            "full_sequence_tokens": record["full_sequence_tokens"],
        }
        for record in longest
    ]
    return {
        "schema_version": 1,
        "execution_time": execution_time
        or datetime.now(timezone.utc).isoformat(),
        "tokenizer": collect_tokenizer_metadata(tokenizer, model_id),
        "template": {
            "name": TEMPLATE_NAME,
            "uses_chat_template": False,
            "add_special_tokens": False,
            "prompt_format": "{prompt}\\n\\nAnswer:\\n",
            "answer_format": "{answer}{eos_token}",
            "description": (
                "The source prompt and answer are preserved verbatim. "
                "The full sequence is tokenized independently."
            ),
        },
        "percentile_method": "linear_interpolation",
        "input_files": {
            split: dict(metadata)
            for split, metadata in sorted(input_files.items())
        },
        "statistics": build_statistics(records),
        "overflow": build_overflow_statistics(records),
        "longest_samples": longest_samples,
        "recommendation": recommend_max_length(records),
    }


def analyze_dataset(
    data_dir: Path,
    tokenizer: Any,
    model_id: str,
    execution_time: str | None = None,
) -> dict[str, Any]:
    records, input_files = load_datasets(data_dir)
    analyzed = analyze_records(records, tokenizer)

    for split in SPLITS:
        path = Path(input_files[split]["path"])
        after_hash = sha256_file(path)
        before_hash = input_files[split]["sha256_before"]
        input_files[split]["sha256_after"] = after_hash
        input_files[split]["unchanged"] = after_hash == before_hash
        if after_hash != before_hash:
            raise RuntimeError(f"input file changed during analysis: {path}")

    return build_report(
        analyzed,
        tokenizer,
        model_id,
        input_files,
        execution_time=execution_time,
    )


def _markdown_stats_table(
    title: str, groups: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| Group | Metric | Count | Min | Mean | Median | P90 | P95 | P99 | Max |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group_name, group_stats in groups.items():
        for metric in LENGTH_FIELDS:
            stats = group_stats[metric]
            lines.append(
                f"| {group_name} | {metric} | {stats['count']} | "
                f"{stats['min']} | {stats['mean']} | {stats['median']} | "
                f"{stats['p90']} | {stats['p95']} | {stats['p99']} | "
                f"{stats['max']} |"
            )
    lines.append("")
    return lines


def render_markdown_report(report: Mapping[str, Any]) -> str:
    tokenizer = report["tokenizer"]
    recommendation = report["recommendation"]
    lines = [
        "# Tokenizer Length Analysis",
        "",
        f"- Model ID: `{tokenizer['model_id']}`",
        f"- Tokenizer: `{tokenizer['tokenizer_class']}`",
        f"- Vocabulary size: `{tokenizer['vocabulary_size']}`",
        f"- EOS: `{tokenizer['eos_token']}` "
        f"(ID `{tokenizer['eos_token_id']}`)",
        f"- PAD: `{tokenizer['pad_token']}` "
        f"(ID `{tokenizer['pad_token_id']}`)",
        f"- BOS: `{tokenizer['bos_token']}` "
        f"(ID `{tokenizer['bos_token_id']}`)",
        f"- Revision: `{tokenizer['revision']}`",
        f"- Transformers: `{tokenizer['transformers_version']}`",
        f"- Executed: `{report['execution_time']}`",
        "",
        "## Candidate Training Format",
        "",
        "```text",
        "{prompt}",
        "",
        "Answer:",
        "{answer}{eos}",
        "```",
        "",
    ]
    lines.extend(
        _markdown_stats_table(
            "Overall Statistics", {"all": report["statistics"]["overall"]}
        )
    )
    lines.extend(
        _markdown_stats_table(
            "Statistics by Split", report["statistics"]["by_split"]
        )
    )
    lines.extend(
        _markdown_stats_table(
            "Statistics by Task Type",
            report["statistics"]["by_task_type"],
        )
    )
    lines.extend(
        [
            "## Recommended max_length",
            "",
            f"- Recommended: `{recommendation['recommended_max_length']}`",
            f"- Overflow count: `{recommendation['overflow_count']}`",
            f"- Overflow ratio: `{recommendation['overflow_ratio']}`",
            f"- Basis: `{', '.join(recommendation['based_on_splits'])}`",
            f"- Reason: {recommendation['reason']}",
        ]
    )
    if recommendation["long_tail_note"]:
        lines.append(f"- Long tail: {recommendation['long_tail_note']}")

    lines.extend(
        [
            "",
            "## Longest 20 Samples",
            "",
            "| ID | Split | Task type | Prompt | Answer | Full sequence |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for sample in report["longest_samples"]:
        lines.append(
            f"| {sample['id']} | {sample['split']} | "
            f"{sample['task_type']} | {sample['prompt_tokens']} | "
            f"{sample['answer_tokens']} | "
            f"{sample['full_sequence_tokens']} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_reports(
    report: Mapping[str, Any], output_json: Path, output_md: Path
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    output_md.write_text(
        render_markdown_report(report), encoding="utf-8", newline="\n"
    )


def load_tokenizer(model_id: str) -> Any:
    from transformers import AutoTokenizer
    from transformers.utils.hub import cached_file

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    config_path = cached_file(
        model_id, "tokenizer_config.json", local_files_only=True
    )
    if config_path:
        path_parts = Path(config_path).parts
        if "snapshots" in path_parts:
            snapshot_index = path_parts.index("snapshots")
            if snapshot_index + 1 < len(path_parts):
                tokenizer._analysis_revision = path_parts[snapshot_index + 1]
    return tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze tokenizer lengths for prepared JSONL splits."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen3-4B-Base",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/processed/tokenizer_report.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("data/processed/tokenizer_report.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = load_tokenizer(args.model_id)
    report = analyze_dataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        model_id=args.model_id,
    )
    write_reports(report, args.output_json, args.output_md)


if __name__ == "__main__":
    main()
