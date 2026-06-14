#!/usr/bin/env python3
"""准备并校验官方的 No Robots 对话分片。"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

from ..common.config import load_yaml_config, validate_stage1_config
from ..common.experiment import sha256_file, write_json, write_jsonl


SOURCE_ID = "HuggingFaceH4/no_robots"
SPLIT_NAMES = {"train": "train", "validation": "test"}
ALLOWED_ROLES = {"system", "user", "assistant"}
NETWORK_URL_VARIABLES = (
    "HF_ENDPOINT",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)
StatusCallback = Callable[[str], None]


def console_status(message: str) -> None:
    timestamp = datetime.now().astimezone().strftime("%H:%M:%S")
    print(
        f"[{timestamp}] [instruction-data] {message}",
        file=sys.stderr,
        flush=True,
    )


def validate_download_environment(
    environment: Mapping[str, str] | None = None,
) -> None:
    """在 HTTPX 发起请求之前拒绝异常的 Hub endpoint 和代理 URL。

    尽早校验可以避免 ``requests.exceptions.InvalidScheme`` 或静默代理
    回退导致的难排查下载卡死。
    """
    values = environment if environment is not None else os.environ
    # 逐变量检查所有可能影响网络下载的环境变量
    for variable in NETWORK_URL_VARIABLES:
        value = values.get(variable)
        if not value:
            continue  # 未设置的变量跳过，不影响正常流程
        # 用 urlsplit 解析 URL，必须同时包含 scheme 和 netloc 才算有效
        parsed = urlsplit(value)
        # HF_ENDPOINT 只允许 http/https，不允许 socks 代理（HF Hub API 不支持 socks）
        # 代理变量额外允许 socks5/socks5h 协议（requests/httpx 底层库支持）
        allowed_schemes = (
            {"http", "https"}
            if variable == "HF_ENDPOINT"
            else {"http", "https", "socks5", "socks5h"}
        )
        # 校验两个条件：1) 协议在白名单中 2) 包含有效的网络地址（netloc 非空）
        # netloc 为空意味着用户只写了协议（如 "http://"）或忘了写协议（如 "127.0.0.1:7890"）
        if parsed.scheme.lower() not in allowed_schemes or not parsed.netloc:
            examples = (
                "https://hf-mirror.com"
                if variable == "HF_ENDPOINT"
                else "http://127.0.0.1:7890"
            )
            raise ValueError(
                f"{variable} must be a complete URL including protocol, "
                f"for example {examples}; got {value!r}"
            )


def _redact_url(value: str) -> str:
    """从 URL 中剥离凭证信息，仅保留 scheme 和 host。"""
    parsed = urlsplit(value)
    hostname = parsed.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = hostname
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    return urlunsplit(
        (parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment)
    )


def describe_download_environment(
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    values = environment if environment is not None else os.environ
    return {
        "endpoint": values.get("HF_ENDPOINT", "https://huggingface.co"),
        "cache": values.get("HF_HOME", "<huggingface default>"),
        "proxies": {
            variable: _redact_url(values[variable])
            for variable in NETWORK_URL_VARIABLES
            if variable != "HF_ENDPOINT" and values.get(variable)
        },
    }


def validate_conversation_record(
    record: Mapping[str, Any], split: str, index: int
) -> None:
    """Validate one no_robots conversation record in place."""
    # 构造一个人类可读的位置字符串，方便错误信息定位具体记录
    location = f"{split}[{index}]"

    # --- 顶层字段校验 ---
    # No Robots 数据集的每条记录必须包含 prompt_id、messages、category 三个字段
    for field in ("prompt_id", "messages", "category"):
        if field not in record:
            raise ValueError(f"{location} missing field: {field}")
    # prompt_id：非空字符串，用作该问题的唯一标识
    if not isinstance(record["prompt_id"], str) or not record["prompt_id"]:
        raise ValueError(f"{location} prompt_id must be a nonempty string")
    # category：非空字符串，标识该问题的类别（如 coding、math 等）
    if not isinstance(record["category"], str) or not record["category"]:
        raise ValueError(f"{location} category must be a nonempty string")
    # messages：非空列表，包含该对话的全部消息轮次
    messages = record["messages"]
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{location} messages must be a nonempty list")

    # --- 逐条消息的角色和内容校验 ---
    previous_role: str | None = None
    assistant_count = 0
    user_count = 0
    for message_index, message in enumerate(messages):
        message_location = f"{location}.messages[{message_index}]"
        # 每条消息必须是 dict/映射类型
        if not isinstance(message, Mapping):
            raise ValueError(f"{message_location} must be an object")
        # 消息对象只能包含 role 和 content 两个字段，防止恶意注入多余字段
        if set(message) != {"role", "content"}:
            raise ValueError(f"{message_location} must contain only role and content")
        role = message["role"]
        content = message["content"]
        # 角色必须是 system/user/assistant 三者之一
        if role not in ALLOWED_ROLES:
            raise ValueError(f"{message_location} unsupported role: {role}")
        # content 必须是字符串类型
        if not isinstance(content, str):
            raise ValueError(f"{message_location} content must be a string")
        # 内容为空的情况：只允许 index=0 处 system 角色的空内容（合法的"空系统提示"）
        # 其他任何位置的内容为空都是无效数据
        if not content and not (message_index == 0 and role == "system"):
            raise ValueError(f"{message_location} content must be a nonempty string")
        # system 角色只能出现在第一条消息（位置 0），这是 OpenAI chat 格式的约定
        if role == "system":
            if message_index != 0:
                raise ValueError(f"{location} system role is only allowed at message 0")
            previous_role = role
            continue  # system 角色跳过后续的用户/助手校验
        # 非 system 消息：第一条必须是 user（用户发起对话）
        if message_index == 0 and role != "user":
            raise ValueError(
                f"{location} invalid role order at message {message_index}: "
                f"expected user, got {role}"
            )
        # 如果有 system 消息（index=0），则 index=1 必须是 user
        # 这确保了消息序列为 [system?, user, assistant, user, assistant, ...]
        if message_index == 1 and previous_role == "system" and role != "user":
            raise ValueError(
                f"{location} invalid role order at message {message_index}: "
                f"expected user, got {role}"
            )
        # user 角色：计数并检查是否有连续两个 user（标准对话应该交替 user/assistant）
        if role == "user":
            user_count += 1
            if previous_role == "user":
                raise ValueError(
                    f"{location} contains consecutive user messages at "
                    f"message {message_index}"
                )
        # assistant 角色：计数
        if role == "assistant":
            assistant_count += 1
        previous_role = role

    # 对话必须至少包含一次用户和一次助手交互
    if user_count == 0 or assistant_count == 0:
        raise ValueError(f"{location} must contain user and assistant messages")
    # 最后一条消息必须是 assistant，这确保对话以模型回复结束（训练目标存在）
    if messages[-1]["role"] != "assistant":
        raise ValueError(f"{location} role order must end with an assistant message")


def prepare_no_robots_records(
    rows: Sequence[Mapping[str, Any]],
    split: str,
    status: StatusCallback | None = None,
) -> list[dict[str, Any]]:
    """校验所有行并附加稳定 ID、来源元数据和类别信息。"""
    if split not in SPLIT_NAMES:
        raise ValueError(f"unsupported Stage 1 split: {split}")
    prepared: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        validate_conversation_record(row, split, index)
        prepared.append(
            {
                "id": f"{SPLIT_NAMES[split]}-{index:05d}",
                "prompt_id": str(row["prompt_id"]),
                "messages": copy.deepcopy(row["messages"]),
                "source": SOURCE_ID,
                "category": row["category"],
                "source_split": SPLIT_NAMES[split],
                "source_row_index": index,
            }
        )
        completed = index + 1
        if status is not None and (completed % 1000 == 0 or completed == len(rows)):
            status(f"Validated {split}: {completed}/{len(rows)} records")
    if not prepared:
        raise ValueError(f"{split} contains no records")
    return prepared


def write_stage1_dataset(
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    dataset_revision: str | None,
    split_fingerprints: Mapping[str, str] | None = None,
    status: StatusCallback | None = None,
) -> dict[str, Any]:
    """校验、存储为 JSONL，并写入 dataset manifest/report。"""
    train = prepare_no_robots_records(train_rows, "train", status=status)
    validation = prepare_no_robots_records(validation_rows, "validation", status=status)
    overlap = {row["id"] for row in train} & {row["id"] for row in validation}
    if overlap:
        raise ValueError(f"train and validation IDs overlap: {sorted(overlap)[:5]}")

    output_dir.mkdir(parents=True, exist_ok=True)
    split_rows = {"train": train, "validation": validation}
    split_paths = {split: output_dir / f"{split}.jsonl" for split in split_rows}
    for split, rows in split_rows.items():
        if status is not None:
            status(f"Writing {split} JSONL: {split_paths[split]}")
        write_jsonl(split_paths[split], rows)

    manifest = {
        "dataset_id": SOURCE_ID,
        "dataset_revision": dataset_revision,
        "splits": {
            split: {
                "source_split": SPLIT_NAMES[split],
                "count": len(rows),
                "sha256": sha256_file(split_paths[split]),
                "ids": [row["id"] for row in rows],
                "arrow_fingerprint": (
                    split_fingerprints.get(split) if split_fingerprints else None
                ),
            }
            for split, rows in split_rows.items()
        },
    }
    report = {
        "status": "ok",
        "total": sum(len(rows) for rows in split_rows.values()),
        "splits": {
            split: {
                "count": len(rows),
                "categories": dict(
                    sorted(Counter(row["category"] for row in rows).items())
                ),
                "message_count": {
                    "min": min(len(row["messages"]) for row in rows),
                    "max": max(len(row["messages"]) for row in rows),
                },
                "consecutive_assistant_records": sum(
                    any(
                        first["role"] == second["role"] == "assistant"
                        for first, second in zip(
                            row["messages"],
                            row["messages"][1:],
                            strict=False,
                        )
                    )
                    for row in rows
                ),
                "duplicate_prompt_id_records": sum(
                    count
                    for count in Counter(row["prompt_id"] for row in rows).values()
                    if count > 1
                ),
                "duplicate_prompt_id_values": sum(
                    count > 1
                    for count in Counter(row["prompt_id"] for row in rows).values()
                ),
                "empty_leading_system_records": sum(
                    bool(row["messages"])
                    and row["messages"][0]["role"] == "system"
                    and row["messages"][0]["content"] == ""
                    for row in rows
                ),
            }
            for split, rows in split_rows.items()
        },
    }
    write_json(output_dir / "dataset_manifest.json", manifest)
    write_json(output_dir / "dataset_report.json", report)
    if status is not None:
        status(f"Wrote dataset manifest and report: {output_dir}")
    return manifest


def prepare_from_hub(
    config: Mapping[str, Any],
    status: StatusCallback = console_status,
) -> dict[str, Any]:
    """Download only the dataset and write validated Stage 1 artifacts.

    revision 解析流程（确保数据可复现）：
    1. 如果配置中显式指定了 dataset_revision，直接使用（即 Git commit SHA）。
    2. 如果未指定（None），调用 HfApi().dataset_info() 获取当前数据集的最新 SHA，
       然后用这个 SHA 作为 revision——后续即使上游数据集更新，
       本次运行也能拉取到完全相同的版本。
    """
    # 第一步：校验网络环境（URL scheme 合法性），提前发现配置错误
    validate_download_environment()
    environment = describe_download_environment()
    status(f"Hub endpoint: {environment['endpoint']}; cache: {environment['cache']}")
    if environment["proxies"]:
        status(f"Proxy configuration: {environment['proxies']}")

    # 延迟导入 HuggingFace 相关库，避免非该流程场景下的无用依赖加载
    from datasets import enable_progress_bars, load_dataset
    from huggingface_hub import HfApi
    from huggingface_hub.utils import enable_progress_bars as enable_hub_progress

    enable_progress_bars()
    enable_hub_progress()

    data_config = config["data"]
    resolved_revision = data_config.get("dataset_revision")

    # revision 解析：如果配置中未指定，从 HuggingFace Hub API 获取最新的 dataset SHA
    # dataset_info().sha 返回的是该数据集当前 HEAD commit 的 SHA，可确保版本固化
    if not resolved_revision:
        status(f"Resolving dataset revision: {data_config['dataset_id']}")
        resolved_revision = HfApi().dataset_info(data_config["dataset_id"]).sha

    # 用确定的 revision 加载数据集
    status(
        f"Loading dataset {data_config['dataset_id']} at revision {resolved_revision}"
    )
    kwargs: dict[str, Any] = {"revision": resolved_revision}
    dataset = load_dataset(data_config["dataset_id"], **kwargs)

    # 从配置中取出 train/validation split 名称
    train_split = data_config["train_split"]
    validation_split = data_config["validation_split"]
    status(
        "Dataset loaded: "
        f"{train_split}={len(dataset[train_split])}, "
        f"{validation_split}={len(dataset[validation_split])}"
    )

    # 传递给 write_stage1_dataset 完成校验、序列化和报告写入
    return write_stage1_dataset(
        list(dataset[train_split]),
        list(dataset[validation_split]),
        Path(data_config["output_dir"]),
        dataset_revision=resolved_revision,
        split_fingerprints={
            "train": dataset[train_split]._fingerprint,
            "validation": dataset[validation_split]._fingerprint,
        },
        status=status,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and validate Stage 1 No Robots data."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage1_no_robots.yaml"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    validate_stage1_config(config)
    manifest = prepare_from_hub(config)
    print(
        "Prepared Stage 1 data: "
        f"train={manifest['splits']['train']['count']}, "
        f"validation={manifest['splits']['validation']['count']}"
    )


if __name__ == "__main__":
    main()
