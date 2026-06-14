#!/usr/bin/env python3
"""Base 与 LoRA 模型变体的统一评估入口。"""

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
    """向 stderr 打印带时间戳的状态消息。"""
    timestamp = datetime.now().astimezone().strftime("%H:%M:%S")
    print(f"[{timestamp}] [evaluate] {message}", file=sys.stderr, flush=True)


@dataclass(frozen=True)
class EvaluationConfig:
    model_id: str  # Hugging Face 模型标识符（如 Qwen/Qwen3-4B-Base）
    model_mode: str  # 加载模式：bf16（BF16精度）/ nf4（4-bit量化）/ lora（LoRA适配器）
    split: str  # 数据集分区：validation 或 test
    data_path: str  # JSONL 格式评估数据文件路径
    output_dir: str  # 评估产物（predictions / metrics / config）输出目录
    cache_dir: str | None = None  # Hugging Face 模型/分词器下载缓存目录（可选）
    adapter_path: str | None = None  # LoRA 适配器权重路径，lora 模式必填
    model_revision: str | None = None  # 基础模型版本标识（Git 哈希/分支），用于可复现加载
    adapter_revision: str | None = None  # LoRA 适配器版本标识（可选）
    max_length: int = DEFAULT_MAX_LENGTH  # 输入 token 最大长度，超过截断
    max_new_tokens: int = 64  # 每次生成的最大新 token 数，限制输出长度
    batch_size: int = 1  # 并行评估的批次大小，控制显存消耗
    limit: int | None = None  # 限制评估样本数（可选，用于快速验证）
    allow_test: bool = False  # 是否允许在受保护的 test 分区上运行（需显式 opt-in）

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
    """拒绝无效 split 并执行 ``--allow-test`` 守卫。"""
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
    """批量生成并将每个预测与标准答案比较。

    ``generate`` 回调接收一批 prompt 字符串，必须返回等长的预测结果序列。
    """
    if not records:
        raise ValueError("cannot evaluate empty records")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    predictions: list[dict[str, Any]] = []
    # 按 batch_size 分批处理所有样本，平衡显存消耗与吞吐效率
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        # 将原始 prompt 包装为统一的评估格式模板（如追加 "Answer:" 提示）
        prompts = [
            format_evaluation_prompt(record["prompt"]) for record in batch
        ]
        # 调用生成回调对整批 prompt 做一次前向传播，利用 batch 并行加速
        generated = list(generate(prompts))
        if len(generated) != len(batch):
            raise RuntimeError(
                "generator returned a different number of predictions "
                f"({len(generated)}) than prompts ({len(batch)})"
            )
        # 逐条对比生成结果与标准答案：根据 task_type 选择对应的解析/比较策略
        for record, prediction_raw in zip(batch, generated, strict=True):
            comparison = evaluate_answer(
                record["task_type"],
                record["answer"],
                prediction_raw,
            )
            # 合并数据集元信息与评估结果为统一预测记录
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
    """将 predictions、error cases、metrics 和 config 写入 *output_dir*。"""
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
    """构建可重现的 run-config 字典用于产物归档。"""
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
    """延迟导入的生成器，支持所有 model mode（bf16 / nf4 / lora）。"""

    def __init__(
        self,
        config: EvaluationConfig,
        status: StatusCallback = console_status,
    ):
        # 延迟导入重型依赖，避免在非评估场景（如数据预处理 CLI）时加载
        status("Importing PyTorch and Transformers")
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        # 检查 CUDA 可用性和 BF16 支持，用于后续 dtype 和 device_map 策略选择
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
        # 左填充（左对齐）：较短序列在左侧补 pad，确保生成的新 token 在序列右侧对齐
        # 这避免了因右侧填充导致生成的 token 被 pad 截断的问题
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("tokenizer has neither PAD nor EOS token")
            # 无专用 pad token 时复用 eos token，避免 attention mask 计算报错
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # device_map="auto" 让 accelerate 自动将模型层分配到可用设备
        model_kwargs: dict[str, Any] = {
            "device_map": "auto",
        }
        if config.model_revision:
            model_kwargs["revision"] = config.model_revision
        if config.cache_dir:
            model_kwargs["cache_dir"] = config.cache_dir
        # 根据 model_mode 决定加载策略：bf16 完整精度 vs nf4 4-bit 量化
        if config.model_mode == "bf16":
            # BF16 模式：直接加载完整 BF16 权重，适合显存充足的场景（如 A100）
            model_kwargs["torch_dtype"] = torch.bfloat16
            load_description = "BF16 Base"
        else:
            # NF4 模式：4-bit 量化 + 双重量化，显著降低显存占用
            # bnb_4bit_use_double_quant=True 对量化常数再做一次量化，进一步压缩
            # bnb_4bit_compute_dtype=BF16 保证前向计算仍用较高精度
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
        # LoRA 模式：在基础模型之上加载微调后的适配器权重
        # 仅推理时使用适配层的增量参数，基础权重保持冻结
        if config.model_mode == "lora":
            # 延迟导入 PEFT，仅在 LoRA 模式需要，避免不必要的依赖
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
        # 切换为评估模式，关闭 dropout 等仅训练时生效的随机行为
        self.model = model.eval()
        self.torch = torch
        self.max_length = config.max_length
        self.max_new_tokens = config.max_new_tokens

    def __call__(self, prompts: Sequence[str]) -> list[str]:
        """对一批 prompt 做 tokenize → generation → decode。"""
        # 分词：同时做填充和截断，确保批次内序列长度一致
        # add_special_tokens=False 因为评估 prompt 已由 format_evaluation_prompt 处理
        inputs = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )
        # 将分词结果（input_ids、attention_mask）移到模型所在设备
        inputs = {
            key: value.to(self.model.device)
            for key, value in inputs.items()
        }
        # 记录输入长度，后续用于从生成输出中截取仅新增的部分
        input_length = inputs["input_ids"].shape[1]
        # inference_mode 等价于 torch.no_grad()，且额外禁用 autograd 开销
        with self.torch.inference_mode():
            # do_sample=False → 贪婪解码（每次选概率最高的 token），保证确定性输出
            # pad_token_id / eos_token_id 确保模型正确处理填充位置和终止条件
            outputs = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        # 只解码新生成的 token 部分（从 input_length 起），跳过输入内容的复读
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
    # 步骤 1：入口日志与分区权限校验（test 分区需显式 --allow-test）
    started = time.monotonic()
    status(
        "Starting evaluation: "
        f"mode={config.model_mode}, split={config.split}, "
        f"limit={config.limit or 'all'}, batch_size={config.batch_size}"
    )
    validate_split_access(config.split, config.allow_test)
    # 步骤 2：加载数据集并计算 SHA256 哈希（用于后续完整性校验）
    data_path = Path(config.data_path)
    status(f"Loading dataset: {data_path}")
    dataset_hash_before = sha256_file(data_path)
    records = read_jsonl(data_path, config.split)
    if config.limit is not None:
        records = records[: config.limit]
    status(f"Loaded {len(records)} {config.split} records")

    # 步骤 3：初始化生成器（延迟导入 + 加载模型权重），执行批量生成与评估
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
    # 步骤 4：二次校验数据集完整性，防止评估过程中文件被外部修改
    dataset_hash_after = sha256_file(data_path)
    if dataset_hash_before != dataset_hash_after:
        raise RuntimeError(f"dataset changed during evaluation: {data_path}")

    # 步骤 5：构建可复现的运行配置快照（含版本号、哈希、时间戳）
    run_config = build_run_config(
        config,
        dataset_hash_before,
        dataset_hash_after,
        len(records),
    )
    # 步骤 6：写入评估产物目录
    # predictions.jsonl / error_cases.jsonl / metrics.json / run_config.json
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
