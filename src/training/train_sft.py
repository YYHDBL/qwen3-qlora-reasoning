#!/usr/bin/env python3
"""YAML-driven BF16 LoRA SFT entry point for Stage 1."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..common.config import (
    apply_overrides,
    load_yaml_config,
    validate_stage1_config,
)
from ..common.experiment import (
    ensure_new_output_dir,
    git_commit,
    package_versions,
    sha256_file,
    write_json,
    write_yaml,
)
from .lora import build_lora_config
from .model_loader import load_bf16_model, load_tokenizer, status


RUN_MODES = ("overfit", "smoke", "formal")
PREFLIGHT_FILES = (
    "dataset_manifest.json",
    "token_report.json",
    "batch_audit.json",
    "overfit_passed.json",
    "adapter_reload.json",
)


def resolve_run_limits(run_mode: str, training: Mapping[str, Any]) -> dict[str, Any]:
    if run_mode == "overfit":
        return {
            "example_limit": int(training.get("overfit_examples", 16)),
            "max_steps": int(training.get("overfit_steps", 40)),
            "num_train_epochs": None,
        }
    if run_mode == "smoke":
        return {
            "example_limit": int(training.get("smoke_examples", 256)),
            "max_steps": int(training.get("smoke_steps", 20)),
            "num_train_epochs": None,
        }
    if run_mode == "formal":
        return {
            "example_limit": None,
            "max_steps": int(training.get("max_steps", -1)),
            "num_train_epochs": training.get("num_train_epochs"),
        }
    raise ValueError(f"unsupported run mode: {run_mode}")


def build_sft_kwargs(
    config: Mapping[str, Any],
    output_dir: Path,
    run_mode: str,
) -> dict[str, Any]:
    training = config["training"]
    limits = resolve_run_limits(run_mode, training)
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir / "checkpoints"),
        "max_length": int(training["max_length"]),
        "per_device_train_batch_size": int(training["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(training["per_device_eval_batch_size"]),
        "gradient_accumulation_steps": int(training["gradient_accumulation_steps"]),
        "gradient_checkpointing": bool(training["gradient_checkpointing"]),
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "assistant_only_loss": bool(training["assistant_only_loss"]),
        "packing": bool(training["packing"]),
        "learning_rate": float(training["learning_rate"]),
        "warmup_ratio": float(training["warmup_ratio"]),
        "lr_scheduler_type": str(training["lr_scheduler_type"]),
        "optim": str(training["optim"]),
        "logging_steps": int(training["logging_steps"]),
        "save_steps": int(training["save_steps"]),
        "eval_steps": int(training["eval_steps"]),
        "eval_strategy": "steps",
        "save_strategy": "steps",
        "bf16": bool(training["bf16"]),
        "tf32": bool(training["tf32"]),
        "seed": int(training["seed"]),
        "data_seed": int(training["seed"]),
        "report_to": ["tensorboard"],
        "logging_dir": str(output_dir / "logs"),
        "remove_unused_columns": True,
        "save_safetensors": True,
        "dataloader_num_workers": 0,
        "max_steps": limits["max_steps"],
    }
    if limits["num_train_epochs"] is not None:
        kwargs["num_train_epochs"] = float(limits["num_train_epochs"])
    return kwargs


def validate_preflight_artifacts(path: Path) -> None:
    missing = [name for name in PREFLIGHT_FILES if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "formal training preflight artifacts are missing: " + ", ".join(missing)
        )


def snapshot_data_artifacts(config: Mapping[str, Any], output_dir: Path) -> None:
    data_dir = Path(config["data"]["output_dir"])
    for name in (
        "dataset_manifest.json",
        "token_report.json",
        "batch_audit.json",
    ):
        source = data_dir / name
        if not source.is_file():
            raise FileNotFoundError(
                f"required data audit artifact is missing: {source}"
            )
        write_json(
            output_dir / name,
            json.loads(source.read_text(encoding="utf-8")),
        )


def _validate_formal_gates(config: Mapping[str, Any]) -> None:
    data_dir = Path(config["data"]["output_dir"])
    output_root = Path(config["experiment"]["output_root"])
    locations = {
        "dataset_manifest.json": data_dir / "dataset_manifest.json",
        "token_report.json": data_dir / "token_report.json",
        "batch_audit.json": data_dir / "batch_audit.json",
        "overfit_passed.json": (output_root / "overfit" / "overfit_passed.json"),
        "adapter_reload.json": (output_root / "smoke" / "adapter_reload.json"),
    }
    missing = [name for name, path in locations.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "formal training preflight artifacts are missing: " + ", ".join(missing)
        )


def _select_dataset(dataset: Any, limit: int | None) -> Any:
    if limit is None or limit >= len(dataset):
        return dataset
    return dataset.select(range(limit))


def _loss_decreased(log_history: list[Mapping[str, Any]]) -> tuple[bool, Any, Any]:
    losses = [float(row["loss"]) for row in log_history if "loss" in row]
    if len(losses) < 2:
        return False, losses[0] if losses else None, None
    return losses[-1] < losses[0] * 0.8, losses[0], losses[-1]


def run_training(
    config: Mapping[str, Any],
    run_mode: str,
    resume_from_checkpoint: Path | None = None,
) -> dict[str, Any]:
    import torch
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    if run_mode == "formal":
        _validate_formal_gates(config)
    output_dir = Path(config["experiment"]["output_root"]) / run_mode
    ensure_new_output_dir(output_dir, resume_from_checkpoint)
    write_yaml(output_dir / "resolved_config.yaml", config)
    snapshot_data_artifacts(config, output_dir)

    data_dir = Path(config["data"]["output_dir"])
    data_paths = {
        split: data_dir / f"{split}.jsonl" for split in ("train", "validation")
    }
    for path in data_paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"prepared dataset is missing: {path}")

    limits = resolve_run_limits(run_mode, config["training"])
    dataset = load_dataset(
        "json",
        data_files={key: str(path) for key, path in data_paths.items()},
    )
    train_dataset = _select_dataset(dataset["train"], limits["example_limit"])
    eval_dataset = _select_dataset(
        dataset["validation"],
        min(len(dataset["validation"]), limits["example_limit"] or 128),
    )

    tokenizer = load_tokenizer(config, for_training=True)
    model = load_bf16_model(config)
    lora_config = build_lora_config(config)

    class TokenCountingSFTTrainer(SFTTrainer):
        supervised_tokens_seen = 0
        total_tokens_seen = 0

        def compute_loss(
            self,
            model: Any,
            inputs: Mapping[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            labels = inputs.get("labels")
            attention_mask = inputs.get("attention_mask")
            if labels is not None and model.training:
                self.supervised_tokens_seen += int(labels.ne(-100).sum().item())
            if attention_mask is not None and model.training:
                self.total_tokens_seen += int(attention_mask.sum().item())
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

    training_args = SFTConfig(**build_sft_kwargs(config, output_dir, run_mode))
    trainer = TokenCountingSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    train_result = trainer.train(
        resume_from_checkpoint=(
            str(resume_from_checkpoint) if resume_from_checkpoint is not None else None
        )
    )
    elapsed = time.monotonic() - started
    eval_metrics = trainer.evaluate()
    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    metrics = {
        "run_mode": run_mode,
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "supervised_tokens_seen": trainer.supervised_tokens_seen,
        "total_tokens_seen": trainer.total_tokens_seen,
        "train": train_result.metrics,
        "validation": eval_metrics,
        "global_step": trainer.state.global_step,
    }
    write_json(output_dir / "train_metrics.json", metrics)
    write_json(output_dir / "eval_metrics.json", eval_metrics)
    write_json(output_dir / "trainer_state.json", asdict(trainer.state))
    write_json(
        output_dir / "environment.json",
        {
            "git_commit": git_commit(),
            "packages": package_versions(
                [
                    "torch",
                    "transformers",
                    "datasets",
                    "peft",
                    "trl",
                    "accelerate",
                ]
            ),
            "gpu_name": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda,
            "bf16_supported": torch.cuda.is_bf16_supported(),
            "model_revision": (
                getattr(model.config, "_commit_hash", None)
                or config["model"].get("revision")
            ),
            "tokenizer_revision": (
                (getattr(tokenizer, "init_kwargs", {}) or {}).get("_commit_hash")
                or getattr(tokenizer, "_commit_hash", None)
            ),
            "dataset_sha256": {
                split: sha256_file(path) for split, path in data_paths.items()
            },
        },
    )

    if run_mode == "overfit":
        passed, first_loss, final_loss = _loss_decreased(trainer.state.log_history)
        overfit_result = {
            "passed": passed,
            "criterion": "final logged loss < 80% of first logged loss",
            "first_loss": first_loss,
            "final_loss": final_loss,
        }
        write_json(output_dir / "overfit_passed.json", overfit_result)
        if not passed:
            raise RuntimeError("overfit gate failed; inspect overfit_passed.json")
    (output_dir / "conclusion.md").write_text(
        "\n".join(
            [
                f"# {config['experiment']['name']} {run_mode}",
                "",
                f"- Global steps: {trainer.state.global_step}",
                (f"- Supervised tokens seen: {trainer.supervised_tokens_seen}"),
                f"- Elapsed seconds: {elapsed:.2f}",
                (f"- Peak GPU memory bytes: {int(torch.cuda.max_memory_allocated())}"),
                "- Review status: pending human review.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Stage 1 BF16 LoRA overfit, smoke, or formal SFT."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage1_no_robots.yaml"),
    )
    parser.add_argument("--mode", choices=RUN_MODES, required=True)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="overrides",
        help="Override a YAML value, for example training.max_length=1024",
    )
    parser.add_argument("--resume-from-checkpoint", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_yaml_config(args.config), args.overrides)
    validate_stage1_config(config)
    status(f"Starting Stage 1 run: mode={args.mode}")
    metrics = run_training(
        config,
        run_mode=args.mode,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
