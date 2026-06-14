#!/usr/bin/env python3
"""Unified evaluation entry point for Base and LoRA model variants."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..common.prompt_format import format_evaluation_prompt
from .analyze_tokens import read_jsonl, sha256_file
from .answer_evaluation import evaluate_answer
from .metrics import compute_metrics


MODEL_MODES = ("bf16", "nf4", "lora")
DEFAULT_MODEL_ID = "Qwen/Qwen3-4B-Base"
DEFAULT_MAX_LENGTH = 512
StatusCallback = Callable[[str], None]
ProgressCallback = Callable[[int, int], None]


def console_status(message: str) -> None:
    timestamp = datetime.now().astimezone().strftime("%H:%M:%S")
    print(f"[{timestamp}] [evaluate] {message}", file=sys.stderr, flush=True)


@dataclass(frozen=True)
class EvaluationConfig:
    model_id: str
    model_mode: str
    split: str
    data_path: str
    output_dir: str
    cache_dir: str | None = None
    adapter_path: str | None = None
    model_revision: str | None = None
    adapter_revision: str | None = None
    max_length: int = DEFAULT_MAX_LENGTH
    max_new_tokens: int = 64
    batch_size: int = 1
    limit: int | None = None
    allow_test: bool = False

    def __post_init__(self) -> None:
        if self.model_mode not in MODEL_MODES:
            raise ValueError(
                f"model mode must be one of: {', '.join(MODEL_MODES)}"
            )
        if self.model_mode == "lora" and not self.adapter_path:
            raise ValueError("LoRA mode requires an adapter path")
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_split_access(split: str, allow_test: bool) -> None:
    if split not in {"validation", "test"}:
        raise ValueError("evaluation split must be validation or test")
    if split == "test" and not allow_test:
        raise ValueError(
            "the protected test split requires explicit --allow-test opt-in"
        )


def evaluate_records(
    records: Sequence[Mapping[str, str]],
    generate: Callable[[Sequence[str]], Sequence[str]],
    batch_size: int,
    progress: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    if not records:
        raise ValueError("cannot evaluate empty records")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    predictions: list[dict[str, Any]] = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        prompts = [
            format_evaluation_prompt(record["prompt"]) for record in batch
        ]
        generated = list(generate(prompts))
        if len(generated) != len(batch):
            raise RuntimeError(
                "generator returned a different number of predictions "
                f"({len(generated)}) than prompts ({len(batch)})"
            )
        for record, prediction_raw in zip(batch, generated, strict=True):
            comparison = evaluate_answer(
                record["task_type"],
                record["answer"],
                prediction_raw,
            )
            predictions.append(
                {
                    "id": record["id"],
                    "split": record["split"],
                    "task_type": record["task_type"],
                    "gold_raw": record["answer"],
                    **comparison,
                }
            )
        if progress is not None:
            progress(min(start + len(batch), len(records)), len(records))
    return predictions


def _write_json(path: Path, value: Any) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary_path.replace(path)


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            )
            handle.write("\n")
    temporary_path.replace(path)


def write_evaluation_artifacts(
    output_dir: Path,
    predictions: Sequence[Mapping[str, Any]],
    run_config: Mapping[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = compute_metrics(predictions)
    error_cases = [
        prediction
        for prediction in predictions
        if not prediction["primary_correct"]
    ]

    _write_jsonl(output_dir / "predictions.jsonl", predictions)
    _write_jsonl(output_dir / "error_cases.jsonl", error_cases)
    _write_json(output_dir / "metrics.json", metrics)
    _write_json(output_dir / "run_config.json", dict(run_config))
    return metrics


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def build_run_config(
    config: EvaluationConfig,
    dataset_sha256_before: str,
    dataset_sha256_after: str,
    evaluated_count: int,
) -> dict[str, Any]:
    return {
        **config.to_dict(),
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "evaluated_count": evaluated_count,
        "dataset_sha256_before": dataset_sha256_before,
        "dataset_sha256_after": dataset_sha256_after,
        "dataset_unchanged": (
            dataset_sha256_before == dataset_sha256_after
        ),
        "prompt_format": "{prompt}\\n\\nAnswer:",
        "generation": {
            "do_sample": False,
            "max_new_tokens": config.max_new_tokens,
        },
        "environment": {
            "git_commit": _git_commit(),
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "accelerate": _package_version("accelerate"),
            "bitsandbytes": _package_version("bitsandbytes"),
            "peft": _package_version("peft"),
        },
    }


class HuggingFaceGenerator:
    """Delayed-import generator shared by all supported model modes."""

    def __init__(
        self,
        config: EvaluationConfig,
        status: StatusCallback = console_status,
    ):
        status("Importing PyTorch and Transformers")
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        if torch.cuda.is_available():
            status(
                "CUDA available: "
                f"{torch.cuda.get_device_name(0)} "
                f"(BF16 supported: {torch.cuda.is_bf16_supported()})"
            )
        else:
            status("CUDA is not available; model loading may fail or be slow")

        tokenizer_kwargs: dict[str, Any] = {}
        if config.model_revision:
            tokenizer_kwargs["revision"] = config.model_revision
        if config.cache_dir:
            tokenizer_kwargs["cache_dir"] = config.cache_dir
        status(
            f"Loading tokenizer: {config.model_id} "
            "(the first run may download files)"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_id, **tokenizer_kwargs
        )
        status(
            f"Tokenizer loaded: {self.tokenizer.__class__.__name__}"
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("tokenizer has neither PAD nor EOS token")
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: dict[str, Any] = {
            "device_map": "auto",
        }
        if config.model_revision:
            model_kwargs["revision"] = config.model_revision
        if config.cache_dir:
            model_kwargs["cache_dir"] = config.cache_dir
        if config.model_mode == "bf16":
            model_kwargs["torch_dtype"] = torch.bfloat16
            load_description = "BF16 Base"
        else:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            load_description = "NF4 4-bit Base"

        status(
            f"Loading model: {config.model_id} as {load_description} "
            "(this can take several minutes on the first run)"
        )
        load_started = time.monotonic()
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id, **model_kwargs
        )
        status(
            f"Base model loaded in {time.monotonic() - load_started:.1f}s"
        )
        if config.model_mode == "lora":
            from peft import PeftModel

            adapter_kwargs: dict[str, Any] = {}
            if config.adapter_revision:
                adapter_kwargs["revision"] = config.adapter_revision
            if config.cache_dir:
                adapter_kwargs["cache_dir"] = config.cache_dir
            status(f"Loading LoRA adapter: {config.adapter_path}")
            model = PeftModel.from_pretrained(
                model, config.adapter_path, **adapter_kwargs
            )
            status("LoRA adapter loaded")
        self.model = model.eval()
        self.torch = torch
        self.max_length = config.max_length
        self.max_new_tokens = config.max_new_tokens

    def __call__(self, prompts: Sequence[str]) -> list[str]:
        inputs = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )
        inputs = {
            key: value.to(self.model.device)
            for key, value in inputs.items()
        }
        input_length = inputs["input_ids"].shape[1]
        with self.torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        return [
            self.tokenizer.decode(
                output[input_length:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for output in outputs
        ]


def run_evaluation(
    config: EvaluationConfig,
    status: StatusCallback = console_status,
) -> dict[str, Any]:
    started = time.monotonic()
    status(
        "Starting evaluation: "
        f"mode={config.model_mode}, split={config.split}, "
        f"limit={config.limit or 'all'}, batch_size={config.batch_size}"
    )
    validate_split_access(config.split, config.allow_test)
    data_path = Path(config.data_path)
    status(f"Loading dataset: {data_path}")
    dataset_hash_before = sha256_file(data_path)
    records = read_jsonl(data_path, config.split)
    if config.limit is not None:
        records = records[: config.limit]
    status(f"Loaded {len(records)} {config.split} records")

    generator = HuggingFaceGenerator(config, status=status)
    status("Starting generation")
    predictions = evaluate_records(
        records,
        generator,
        batch_size=config.batch_size,
        progress=lambda completed, total: status(
            f"Generated {completed}/{total} samples "
            f"({completed / total:.1%})"
        ),
    )
    dataset_hash_after = sha256_file(data_path)
    if dataset_hash_before != dataset_hash_after:
        raise RuntimeError(f"dataset changed during evaluation: {data_path}")

    run_config = build_run_config(
        config,
        dataset_hash_before,
        dataset_hash_after,
        len(records),
    )
    status(f"Writing evaluation artifacts: {config.output_dir}")
    metrics = write_evaluation_artifacts(
        Path(config.output_dir), predictions, run_config
    )
    status(
        f"Evaluation complete: {len(records)} samples in "
        f"{time.monotonic() - started:.1f}s; "
        f"primary_accuracy={metrics['overall']['primary_accuracy']:.4f}"
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3 Base or LoRA models with shared metrics."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--model-mode", choices=MODEL_MODES, default="bf16"
    )
    parser.add_argument("--adapter-path")
    parser.add_argument("--model-revision")
    parser.add_argument("--adapter-revision")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Store Hugging Face downloads in this directory",
    )
    parser.add_argument(
        "--split", choices=("validation", "test"), default="validation"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--allow-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = args.data_path or args.data_dir / f"{args.split}.jsonl"
    config = EvaluationConfig(
        model_id=args.model_id,
        model_mode=args.model_mode,
        split=args.split,
        data_path=str(data_path),
        output_dir=str(args.output_dir),
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
        adapter_path=args.adapter_path,
        model_revision=args.model_revision,
        adapter_revision=args.adapter_revision,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        limit=args.limit,
        allow_test=args.allow_test,
    )
    metrics = run_evaluation(config)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
