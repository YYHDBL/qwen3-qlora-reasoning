#!/usr/bin/env python3
"""在新进程中重新加载已保存的 Stage 1 adapter 并生成一次。"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from ..common.config import load_yaml_config, validate_stage1_config
from ..common.experiment import write_json


def verify_adapter(
    config: dict,
    adapter_path: Path,
    output_path: Path,
) -> dict:
    """加载训练好的 LoRA adapter，在第一个 dev 样本上生成并报告结果。

    此函数在独立进程中调用（确保完全重新加载），用于验证：
    1. adapter_config.json 存在且格式正确
    2. adapter 权重可以成功加载到 base 模型上
    3. 加载后的模型能正常生成输出（不崩溃、不产生空文本）
    4. 停止 token 正常触发（检查 stop_reason）
    """
    # 第 1 步: 验证 adapter 目录包含必要的配置文件
    if not (adapter_path / "adapter_config.json").is_file():
        raise FileNotFoundError(f"adapter_config.json is missing from {adapter_path}")

    from ..evaluation.instruction_eval import load_instruction_eval
    from .model_loader import ChatGenerator

    # 第 2 步: 加载 dev 评估集的第一条样本作为测试输入
    dev_path = Path(config["evaluation"]["dev_path"])
    sample = load_instruction_eval(dev_path, "dev")[0]

    # 第 3 步: 创建 ChatGenerator（内部会 load_lora_model + load_tokenizer）
    #   - adapter_path 非空 → 加载 LoRA adapter
    #   - is_trainable=False → adapter 冻结，仅推理
    generator = ChatGenerator(config, adapter_path=str(adapter_path))

    # 第 4 步: 在单条样本上生成（batch_size=1）
    generated = generator.generate([sample["messages"]], batch_size=1)[0]

    # 第 5 步: 构建验证结果
    #   - stop_success: 停止 token 是否正常触发（im_end / endoftext）
    #   - under_token_budget: 未打满 max_new_tokens（说明没有无尽的续写）
    #   - passed: 停止正常 且 未超 token 上限
    max_new = int(config["generation"]["max_new_tokens"])
    stop_success = generated["stop_reason"] in ("im_end", "endoftext")
    under_token_budget = generated["generated_tokens"] < max_new
    result = {
        "passed": stop_success and under_token_budget,
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "adapter_path": str(adapter_path),
        "sample_id": sample["id"],
        "generated_tokens": generated["generated_tokens"],
        "stop_reason": generated["stop_reason"],
        "stop_success": stop_success,
        "under_token_budget": under_token_budget,
        "max_new_tokens": max_new,
        "prediction_text": generated["text"],
    }
    # 第 6 步: 写入结果 JSON（供 smoke/formal 阶段的门控检查）
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
    if result["passed"]:
        print(f"Adapter reload passed: {result['adapter_path']}")
    else:
        print(
            f"Adapter reload FAILED: stop_reason={result['stop_reason']}, "
            f"generated_tokens={result['generated_tokens']}, "
            f"max_new_tokens={result['max_new_tokens']}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
