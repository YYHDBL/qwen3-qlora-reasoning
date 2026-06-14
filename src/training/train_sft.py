#!/usr/bin/env python3
"""YAML 驱动的 BF16 LoRA SFT 入口，用于 Stage 1。"""

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

try:
    import swanlab
    from swanlab.integration.transformers import SwanLabCallback
except ImportError:
    swanlab = None
    SwanLabCallback = None


RUN_MODES = ("overfit", "smoke", "formal")
PREFLIGHT_FILES = (
    "dataset_manifest.json",
    "token_report.json",
    "batch_audit.json",
    "overfit_passed.json",
    "adapter_reload.json",
)


def resolve_run_limits(run_mode: str, training: Mapping[str, Any]) -> dict[str, Any]:
    # overfit 模式：用极少量样本（默认16条）验证模型能"背诵"数据，确认训练管线无 bug
    # 少量 steps（默认40）足够让模型在小数据上快速收敛到极低 loss
    if run_mode == "overfit":
        return {
            "example_limit": int(training.get("overfit_examples", 16)),
            "max_steps": int(training.get("overfit_steps", 40)),
            "num_train_epochs": None,  # 由 max_steps 控制停止，不设置 epoch 数
        }
    # smoke 模式：快速冒烟测试，用中等样本数（默认256）跑少量 steps（默认20）
    # 验证模型在更多数据上训练不崩溃，训练速度可接受
    if run_mode == "smoke":
        return {
            "example_limit": int(training.get("smoke_examples", 256)),
            "max_steps": int(training.get("smoke_steps", 20)),
            "num_train_epochs": None,
        }
    # formal 模式：正式训练，不限制样本数（None 表示全量数据）
    # max_steps=-1 表示由 epoch 数决定训练长度
    if run_mode == "formal":
        return {
            "example_limit": None,
            "max_steps": int(training.get("max_steps", -1)),
            "num_train_epochs": training.get("num_train_epochs"),
        }
    raise ValueError(f"unsupported run mode: {run_mode}")


def _init_swanlab(
    config: Mapping[str, Any], run_mode: str, output_dir: Path
) -> None:
    """根据 YAML 配置启动 SwanLab 实验。

    若 config 中设置了 ``swanlab_api_key`` 则用它登录，否则 SDK 会回退
    到 ``SWANLAB_API_KEY`` 环境变量。日志写入 ``<output_dir>/swanlab/``。
    """
    if swanlab is None:
        return
    experiment = config["experiment"]
    api_key = experiment.get("swanlab_api_key")
    if api_key and isinstance(api_key, str):
        swanlab.login(api_key=api_key)
    swanlab.init(
        project=experiment["swanlab_project"],
        workspace=experiment.get("swanlab_workspace"),
        experiment_name=f"{experiment['name']}-{run_mode}",
        logdir=str(output_dir / "swanlab"),
    )


def _add_swanlab_callback(
    trainer: Any, config: Mapping[str, Any], run_mode: str
) -> None:
    """向 HuggingFace trainer 附加 ``SwanLabCallback``。

    callback 自动记录训练/评估指标、系统指标（GPU、内存）以及解析后的配置。
    """
    if SwanLabCallback is None:
        return
    experiment = config["experiment"]
    trainer.add_callback(
        SwanLabCallback(
            project=experiment["swanlab_project"],
            workspace=experiment.get("swanlab_workspace"),
            experiment_name=f"{experiment['name']}-{run_mode}",
            log_config=True,
        )
    )


def build_sft_kwargs(
    config: Mapping[str, Any],
    output_dir: Path,
    run_mode: str,
) -> dict[str, Any]:
    """根据 YAML 配置构建 TRL SFTConfig 所需的 kwargs 字典。

    每个参数的设计依据：
    - max_length: 训练时序列最大 token 数，超出会被截断
    - per_device_train_batch_size: 单卡训练 batch，需配合 gradient_accumulation_steps 达到有效 batch
    - per_device_eval_batch_size: 评估 batch，可适当大于训练 batch（不存梯度，显存压力小）
    - gradient_accumulation_steps: 梯度累积步数，有效 batch = batch_size * accumulation_steps * num_gpus
    - gradient_checkpointing: 用计算换显存，use_reentrant=False 是 PyTorch 2.0+ 推荐的非可重入模式
    - assistant_only_loss: 只在 assistant 回复部分计算 loss，忽略 user/system prompt 的 token
    - packing: 将多条短序列打包成一个训练样本，提高 GPU 利用率，但需要 attention mask 支持
    - learning_rate: LoRA 微调典型值 1e-4 ~ 5e-5
    - warmup_ratio: 0.1 表示前 10% 的 steps 学习率从 0 线性增长到目标值
    - lr_scheduler_type: cosine 在训练末期逐步降低学习率，有助于稳定收敛
    - optim: adamw_torch 无需额外安装 apex，在 BF16 下与 adamw_8bit 效果接近
    - logging_steps / save_steps / eval_steps: 控制日志频率、保存频率、评估频率
    - bf16: 使用 BF16 混合精度训练，显存占用约为 FP32 的一半
    - tf32: 在 Ampere+ GPU 上启用 TF32 matmul 加速
    - seed / data_seed: 同时设置模型和数据 shuffle 的随机种子，保证可复现性
    - report_to: 除了 swanlab 外也保留 tensorboard 作为备用日志
    - remove_unused_columns: 移除数据集中 trainer 不使用的列，避免 tokenizer 报错
    - dataloader_num_workers: 0 避免多进程数据加载（tokenized 数据集已预处理完，多进程无收益且可能引发死锁）
    - max_steps vs num_train_epochs: 互斥，优先用 max_steps（overfit/smoke），formal 用 epoch 控制
    """
    training = config["training"]
    limits = resolve_run_limits(run_mode, training)
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir / "checkpoints"),
        # 序列最大长度，超过的样本在 label_audit 阶段已被剔除
        "max_length": int(training["max_length"]),
        # 单卡每步训练的样本数
        "per_device_train_batch_size": int(training["per_device_train_batch_size"]),
        # 单卡每步评估的样本数
        "per_device_eval_batch_size": int(training["per_device_eval_batch_size"]),
        # 梯度累积：多步小 batch 累积后再更新参数，等效于增大 batch size
        "gradient_accumulation_steps": int(training["gradient_accumulation_steps"]),
        # 梯度检查点：前向时不保存中间激活，反向时重新计算，牺牲 20% 速度换显存
        "gradient_checkpointing": bool(training["gradient_checkpointing"]),
        # use_reentrant=False: PyTorch 原生实现的 gradient checkpointing，更安全
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        # 只在 assistant 的 generation block 内计算 loss
        "assistant_only_loss": bool(training["assistant_only_loss"]),
        # 打包多条短样本到同一序列
        "packing": bool(training["packing"]),
        # LoRA 典型学习率 1e-4，比全量微调高一个数量级
        "learning_rate": float(training["learning_rate"]),
        # 前 10% steps 做学习率 warmup，避免初始阶段不稳定
        "warmup_ratio": float(training["warmup_ratio"]),
        # cosine schedule: 学习率按余弦曲线衰减到接近 0
        "lr_scheduler_type": str(training["lr_scheduler_type"]),
        # PyTorch 原生的 AdamW，避免依赖第三方优化器
        "optim": str(training["optim"]),
        "logging_steps": int(training["logging_steps"]),
        "save_steps": int(training["save_steps"]),
        "eval_steps": int(training["eval_steps"]),
        # 按 steps 触发评估和保存，而不是 epoch
        "eval_strategy": "steps",
        "save_strategy": "steps",
        # BF16 混合精度：计算在 BF16 下进行，权重存储和更新在 FP32
        "bf16": bool(training["bf16"]),
        # TF32: NVIDIA Ampere+ GPU 上的 19 位浮点格式，比 FP32 快但精度略低
        "tf32": bool(training["tf32"]),
        # 随机种子同时用于模型初始化和数据 shuffle
        "seed": int(training["seed"]),
        "data_seed": int(training["seed"]),
        # 同时输出到 tensorboard（swanlab 通过 callback 单独添加）
        "report_to": ["tensorboard"],
        "logging_dir": str(output_dir / "logs"),
        # 移除未使用的列，避免 tokenizer 处理多余字段时报 key 不存在错误
        "remove_unused_columns": True,
        # 单进程数据加载：tokenized 数据集已预处理，多进程无收益
        "dataloader_num_workers": 0,
        # max_steps: 训练步数上限（overfit/smoke 模式用 steps 控制停止）
        "max_steps": limits["max_steps"],
    }
    # 仅 formal 模式设置 num_train_epochs，与 max_steps=-1 搭配使用
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


def load_audit_exclusions(
    data_dir: Path,
    expected_max_length: int,
) -> dict[str, set[str]]:
    report_path = data_dir / "token_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"token audit report is missing: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    audited_max_length = report.get("max_length")
    if audited_max_length != expected_max_length:
        raise ValueError(
            "token audit max_length does not match training config: "
            f"report={audited_max_length}, config={expected_max_length}; "
            "rerun src.training.label_audit"
        )
    splits = report.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("token audit report is missing split statistics")
    return {
        split: set(splits.get(split, {}).get("excluded_ids", []))
        for split in ("train", "validation")
    }


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
    # 如果没有限制 (None) 或限制数大于等于数据集大小，返回全量
    # 否则只取前 limit 条（对 overfit/smoke 模式限制样本数）
    if limit is None or limit >= len(dataset):
        return dataset
    return dataset.select(range(limit))


def _loss_decreased(log_history: list[Mapping[str, Any]]) -> tuple[bool, Any, Any]:
    # 从 trainer log_history 中提取所有 loss 记录（每次 logging_steps 产生一条）
    losses = [float(row["loss"]) for row in log_history if "loss" in row]
    # 至少需要 2 个 loss 数据点才能判断下降趋势
    if len(losses) < 2:
        return False, losses[0] if losses else None, None
    # overfit 门控条件：最终 loss < 初始 loss * 0.8（即下降了 20% 以上）
    # 只有 loss 显著下降才说明模型确实在小数据集上"学到"了，而非随机猜测
    return losses[-1] < losses[0] * 0.8, losses[0], losses[-1]


def run_training(
    config: Mapping[str, Any],
    run_mode: str,
    resume_from_checkpoint: Path | None = None,
) -> dict[str, Any]:
    import torch
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    # ── 第 0 步：formal 模式需要前置关卡（数据审计、overfit、smoke 都通过）──
    if run_mode == "formal":
        _validate_formal_gates(config)
    # ── 初始化输出目录与环境 ──
    output_dir = Path(config["experiment"]["output_root"]) / run_mode
    # ensure_new_output_dir：如果目录已存在且非 resume 模式，检查是否为空；resume 时不清理
    ensure_new_output_dir(output_dir, resume_from_checkpoint)
    # 将运行时解析后的完整配置写入 JSON，便于复现
    write_yaml(output_dir / "resolved_config.yaml", config)
    # 初始化 swanlab 实验跟踪
    _init_swanlab(config, run_mode, output_dir)
    # 将数据审计产物（manifest、token_report、batch_audit）快照到训练输出目录
    snapshot_data_artifacts(config, output_dir)

    # ── 第 1 步：数据加载 ──
    data_dir = Path(config["data"]["output_dir"])
    # 训练/验证数据来自 label_audit 阶段输出的 JSONL 文件
    data_paths = {
        split: data_dir / f"{split}.jsonl" for split in ("train", "validation")
    }
    for path in data_paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"prepared dataset is missing: {path}")

    limits = resolve_run_limits(run_mode, config["training"])
    # 用 HuggingFace datasets 加载 JSONL，自动创建 train/validation 两个 split
    dataset = load_dataset(
        "json",
        data_files={key: str(path) for key, path in data_paths.items()},
    )

    # ── 第 2 步：过滤超长样本 ──
    # 从 token_report.json 中读取被标记为超出 max_length 的样本 ID，在训练前剔除
    exclusions = load_audit_exclusions(
        data_dir,
        expected_max_length=int(config["training"]["max_length"]),
    )
    filtered_dataset = {}
    for split in ("train", "validation"):
        split_dataset = dataset[split]
        excluded_ids = exclusions[split]
        if excluded_ids:
            # 使用 filter 而非 select，因为 exclude 集合可能来自不同顺序的索引
            split_dataset = split_dataset.filter(
                lambda row: row["id"] not in excluded_ids,
                desc=f"Excluding {split} records above max_length",
            )
        filtered_dataset[split] = split_dataset
        status(
            f"{split}: retained {len(split_dataset)}/{len(dataset[split])} "
            f"records after token audit"
        )

    # ── 第 3 步：应用模式限制（overfit/smoke 截断样本数）──
    train_dataset = _select_dataset(
        filtered_dataset["train"],
        limits["example_limit"],
    )
    eval_dataset = _select_dataset(
        filtered_dataset["validation"],
        # 评估集也限制数量，避免评估耗时过长；eval limit 取 validation 大小和 128 的较小值
        min(
            len(filtered_dataset["validation"]),
            limits["example_limit"] or 128,
        ),
    )

    # ── 第 4 步：模型加载 ──
    # tokenizer: padding_side="right"（训练用右侧 padding，loss 计算不受影响）
    tokenizer = load_tokenizer(config, for_training=True)
    # 以 BF16 精度加载 base 模型，low_cpu_mem_usage=True 避免 CPU 内存双份拷贝
    model = load_bf16_model(config)
    # 构建 LoRA 配置（r、alpha、dropout、target_modules 等）
    lora_config = build_lora_config(config)

    # ── 第 4.5 步：自定义 Trainer（统计监督/总 token 数）──
    class TokenCountingSFTTrainer(SFTTrainer):
        """``trl.SFTTrainer`` 的扩展，统计监督 token 数与总 token 数。

        计数在每次 ``compute_loss`` 调用中累加，最终写入 metrics JSON，
        便于报告运行时监督/总 token 比率。
        """
        supervised_tokens_seen = 0
        total_tokens_seen = 0

        def compute_loss(
            self,
            model: Any,
            inputs: Mapping[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            # labels != -100 表示该位置参与 loss 计算（即 assistant 的 token）
            labels = inputs.get("labels")
            attention_mask = inputs.get("attention_mask")
            # 只在训练模式（非 eval 模式）下统计，避免评估时重复计数
            if labels is not None and model.training:
                # labels.ne(-100): 布尔张量，True 表示该位置是监督 token
                self.supervised_tokens_seen += int(labels.ne(-100).sum().item())
            if attention_mask is not None and model.training:
                # attention_mask.sum(): 有效 token 总数（非 padding 部分）
                self.total_tokens_seen += int(attention_mask.sum().item())
            # 调用父类的 compute_loss 完成实际的 loss 计算
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

    # ── 第 5 步：创建 Trainer ──
    training_args = SFTConfig(**build_sft_kwargs(config, output_dir, run_mode))
    trainer = TokenCountingSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    # 附加 swanlab 回调，自动记录训练/评估/系统指标
    _add_swanlab_callback(trainer, config, run_mode)

    # ── 第 6 步：训练 ──
    # 重置 GPU 峰值内存统计，用于事后报告
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    train_result = trainer.train(
        resume_from_checkpoint=(
            str(resume_from_checkpoint) if resume_from_checkpoint is not None else None
        )
    )
    elapsed = time.monotonic() - started

    # ── 第 7 步：评估 ──
    eval_metrics = trainer.evaluate()

    # ── 第 8 步：保存 ──
    # 保存 LoRA adapter 权重和 tokenizer（含 chat_template）
    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    # ── 第 9 步：收集并写入所有指标 ──
    metrics = {
        "run_mode": run_mode,
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        # 来自自定义 Trainer 的 token 统计
        "supervised_tokens_seen": trainer.supervised_tokens_seen,
        "total_tokens_seen": trainer.total_tokens_seen,
        "train": train_result.metrics,
        "validation": eval_metrics,
        "global_step": trainer.state.global_step,
    }
    # 同步写入 swanlab（如已启用）
    if swanlab is not None:
        swanlab.log(
            {
                "train/loss": train_result.metrics.get("train_loss"),
                "train/runtime": train_result.metrics.get("train_runtime"),
                "eval/loss": eval_metrics.get("eval_loss"),
                "peak_gpu_memory_bytes": metrics["peak_gpu_memory_bytes"],
                "supervised_tokens_seen": metrics["supervised_tokens_seen"],
                "total_tokens_seen": metrics["total_tokens_seen"],
            }
        )
    # 保存训练指标、评估指标、trainer 完整状态
    write_json(output_dir / "train_metrics.json", metrics)
    write_json(output_dir / "eval_metrics.json", eval_metrics)
    write_json(output_dir / "trainer_state.json", asdict(trainer.state))
    # 保存环境快照：Git commit、依赖版本、GPU 信息、数据集哈希，用于事后复现
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

    # ── 第 10 步：overfit 模式专用门控检查 ──
    if run_mode == "overfit":
        # 检查 loss 是否从初始值下降到 80% 以下
        passed, first_loss, final_loss = _loss_decreased(trainer.state.log_history)
        overfit_result = {
            "passed": passed,
            "criterion": "final logged loss < 80% of first logged loss",
            "first_loss": first_loss,
            "final_loss": final_loss,
        }
        write_json(output_dir / "overfit_passed.json", overfit_result)
        # 未通过则抛异常，阻止后续 smoke/formal 运行
        if not passed:
            raise RuntimeError("overfit gate failed; inspect overfit_passed.json")
    # 写入可读的总结 Markdown
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
    # swanlab.finish: 结束本次实验记录，将缓冲数据上传到 swanlab 云端
    # 必须在函数退出前调用，否则部分日志可能丢失
    if swanlab is not None:
        swanlab.finish()
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
