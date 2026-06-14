"""确定性分层训练/验证/测试集划分。

按任务类型使用最大余数法（Largest-Remainder）分配样本，
保证三个 split 中各类别比例一致。
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from statistics import mean, median
from typing import Iterable, Mapping, Sequence

from .classifier import TASK_ORDER


# 使用固定种子保证每次运行的分片结果可复现
DEFAULT_SEED = 42
# 80/10/10 的划分比例：80% 训练集用于模型参数更新，
# 10% 验证集用于超参调优和早停判断，10% 测试集用于最终评估。
# 对于小规模推理数据集，10% 的验证/测试集足够评估泛化能力，
# 同时最大化训练数据量。
SPLIT_RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}


def _tie_break(seed: int, name: str) -> str:
    """通过带种子的 SHA-256 进行确定性平局裁决。"""
    value = f"{seed}:{name}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _apportion(
    counts: Mapping[str, int],
    ratio: float,
    target_total: int,
    seed: int,
) -> dict[str, int]:
    """使用最大余数法（Hare 配额）按任务分配槽位。

    最大余数法（Largest-Remainder Method）是一种比例分配算法，
    保证每种任务类型在 split 中的占比尽可能接近全局占比。
    """
    # 第一步：计算每种任务的"理想配额" = 全局数量 × 目标比例
    # 例如：bit_manipulation 有 100 条，ratio=0.8，理想配额 = 80.0
    raw = {task: count * ratio for task, count in counts.items()}

    # 第二步：先取 floor（向下取整），保证每种任务至少获得整数个槽位
    allocation = {task: math.floor(value) for task, value in raw.items()}

    # 第三步：计算 floor 分配后还剩下多少槽位需要分配
    # 例如：floor 分配了 78 个，target_total=80，则 remaining=2
    remaining = target_total - sum(allocation.values())

    # 第四步：按"小数余数"降序排序，余数大的任务优先获得额外槽位
    # 排序键 = (负的余数, tie_break哈希)，负值是为了降序（Python 默认升序）
    # tie_break 确保余数相同时有确定性的次要排序
    ranked_tasks = sorted(
        counts,
        key=lambda task: (
            -(raw[task] - math.floor(raw[task])),
            _tie_break(seed, task),
        ),
    )

    # 第五步：将剩余槽位按排序顺序逐个分配给余数最大的任务
    for task in ranked_tasks[:remaining]:
        allocation[task] += 1

    return allocation


def allocate_split_quotas(
    task_counts: Mapping[str, int],
    ratios: Mapping[str, float] = SPLIT_RATIOS,
    seed: int = DEFAULT_SEED,
) -> dict[str, dict[str, int]]:
    """Allocate exact global split totals while preserving task proportions.

    最大余数法分配流程（Largest-Remainder Method）：
    1. 根据 ratios 计算 train/validation 的全局 target_total。
    2. 对 train/validation 分别调用 _apportion 做比例分配。
    3. test 用减法获得剩余样本（保证所有样本都被分配，无遗漏无重复）。
    4. 如果 test 配额为负，说明 train+validation 分配溢出，立即报错。
    """
    # 防御性校验：ratios 必须恰好包含 train/validation/test 三个键
    if set(ratios) != {"train", "validation", "test"}:
        raise ValueError("ratios must contain train, validation, and test")
    # 比例总和必须为 1.0（用 math.isclose 处理浮点误差）
    if not math.isclose(sum(ratios.values()), 1.0):
        raise ValueError("split ratios must sum to 1")

    total = sum(task_counts.values())
    # 先取整计算 train 和 validation 的目标数
    train_total = round(total * ratios["train"])
    validation_total = round(total * ratios["validation"])

    # 用最大余数法分别分配 train 和 validation
    # seed 不同避免两个 split 的 tie-break 产生相同排序
    train = _apportion(
        task_counts, ratios["train"], train_total, seed=seed
    )
    validation = _apportion(
        task_counts,
        ratios["validation"],
        validation_total,
        seed=seed + 1,
    )
    # test 通过减法得到：所有样本中扣除 train 和 validation 后的剩余
    # 这保证了总数精确匹配，且不会有样本被遗漏
    test = {
        task: task_counts[task] - train[task] - validation[task]
        for task in task_counts
    }
    # 安全检查：由于 round 可能导致 train+validation > total
    # 此时某些任务的 test 配额会为负，必须中止
    if any(count < 0 for count in test.values()):
        raise ValueError("split allocation produced a negative test quota")

    return {"train": train, "validation": validation, "test": test}


def _build_splits(
    rows: Sequence[dict[str, str]],
    seed: int,
) -> tuple[dict[str, list[dict[str, str]]], dict[str, dict[str, int]]]:
    """按任务 shuffle，按配额分配，最后 shuffle 每个 split。

    分配流水线（三层 shuffle 保证随机性且可复现）：
    1. 按 task_type 分组 → 为后续按比例分配做准备。
    2. 调用 allocate_split_quotas 计算每个 split 每类任务应分配的数量。
    3. 每类任务内部 shuffle（种子 = seed + task_index），然后按配额切片分配到各 split。
    4. 每个 split 整体 shuffle（种子 = seed + 100 + split_index），
       打乱不同任务类型的顺序，避免训练时看到固定任务顺序。
    """
    # --- 第 1 步：按 task_type 分组 ---
    rows_by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_task[row["task_type"]].append(row)

    # 只用实际存在样本的任务类型计算配额
    task_counts = {
        task: len(rows_by_task[task])
        for task in TASK_ORDER
        if rows_by_task[task]
    }
    # --- 第 2 步：计算配额 ---
    quotas = allocate_split_quotas(task_counts, seed=seed)

    splits: dict[str, list[dict[str, str]]] = {
        split: [] for split in SPLIT_RATIOS
    }

    # --- 第 3 步：按任务 shuffle 后切片分配 ---
    for task_index, task in enumerate(TASK_ORDER):
        task_rows = list(rows_by_task[task])
        # 每种任务的 shuffle 种子 = 基础种子 + 任务在 TASK_ORDER 中的索引
        # 这样保证了不同任务间的 shuffle 独立，但整体可复现
        random.Random(seed + task_index).shuffle(task_rows)

        # 按配额将 shuffle 后的列表切片分配给三个 split
        # train 拿到前 train_end 个，validation 拿中间，test 拿末尾
        train_end = quotas["train"].get(task, 0)
        validation_end = train_end + quotas["validation"].get(task, 0)
        splits["train"].extend(task_rows[:train_end])
        splits["validation"].extend(task_rows[train_end:validation_end])
        splits["test"].extend(task_rows[validation_end:])

    # --- 第 4 步：每个 split 整体 shuffle ---
    # 最终 shuffle 打乱了不同任务类型的顺序，避免模型在训练时习得任务顺序偏见
    # 种子 = seed + 100 + split_index，偏移 100 避免与第 3 步的种子冲突
    for split_index, split in enumerate(SPLIT_RATIOS):
        random.Random(seed + 100 + split_index).shuffle(splits[split])

    return splits, quotas


def _percentile(sorted_values: Sequence[int], percentile: float) -> int:
    """对已排序序列计算最近秩百分位。"""
    if not sorted_values:
        return 0
    index = min(
        len(sorted_values) - 1,
        math.ceil(percentile * len(sorted_values)) - 1,
    )
    return sorted_values[index]


def _length_stats(values: Iterable[str]) -> dict[str, float | int]:
    """返回字符串集合的 min/max/mean/median/p90/p99。"""
    lengths = sorted(len(value) for value in values)
    if not lengths:
        return {
            "min": 0,
            "max": 0,
            "mean": 0,
            "median": 0,
            "p90": 0,
            "p99": 0,
        }
    return {
        "min": lengths[0],
        "max": lengths[-1],
        "mean": round(mean(lengths), 2),
        "median": median(lengths),
        "p90": _percentile(lengths, 0.90),
        "p99": _percentile(lengths, 0.99),
    }
