#!/usr/bin/env python3
"""Reload a saved Stage 1 adapter in a fresh process and generate once."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from ..common.config import load_yaml_config, validate_stage1_config
from ..common.experiment import write_json


def verify_adapter(
    config: dict,
    adapter_path: Path,
    output_path: Path,
) -> dict:
    if not (adapter_path / "adapter_config.json").is_file():
        raise FileNotFoundError(f"adapter_config.json is missing from {adapter_path}")
    from ..evaluation.instruction_eval import load_instruction_eval
    from .model_loader import ChatGenerator

    dev_path = Path(config["evaluation"]["dev_path"])
    sample = load_instruction_eval(dev_path, "dev")[0]
    generator = ChatGenerator(config, adapter_path=str(adapter_path))
    generated = generator.generate([sample["messages"]], batch_size=1)[0]
    result = {
        "passed": True,
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "adapter_path": str(adapter_path),
        "sample_id": sample["id"],
        "generated_tokens": generated["generated_tokens"],
        "stop_reason": generated["stop_reason"],
    }
    write_json(output_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reload and verify a Stage 1 LoRA adapter."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage1_no_robots.yaml"),
    )
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    validate_stage1_config(config)
    result = verify_adapter(config, args.adapter_path, args.output)
    print(f"Adapter reload passed: {result['adapter_path']}")


if __name__ == "__main__":
    main()
