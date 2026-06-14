"""按任务感知的答案提取与比较。"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any

from ..data_processing.classifier import (
    BINARY_ANSWER_RE,
    GRAVITY_ANSWER_RE,
    ROMAN_RE,
    SYMBOL_TOKEN_RE,
    TASK_ORDER,
    UNIT_ANSWER_RE,
    WORDS_RE,
)


ANSWER_LABEL_RE = re.compile(
    r"^(?:final answer|answer|the answer is):[ \t]*(.*)$",
    flags=re.IGNORECASE,
)
DECIMAL_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _extract_candidate(raw_prediction: str) -> tuple[str | None, bool, str]:
    """从原始生成文本中提取候选答案。

    返回 ``(candidate, clean_output, method)``。
    *clean_output* 在输出为单行标记或纯文本时为 ``True``；
    *method* 描述候选提取方式（``"plain"``、``"labeled"``、``"multiline"`` 等）。
    """
    # 第一步：统一换行符为 \n，消除 Windows（\r\n）和 macOS（\r）差异
    normalized_newlines = raw_prediction.replace("\r\n", "\n").replace(
        "\r", "\n"
    )
    # 第二步：提取所有非空行，去除首尾空白字符
    nonempty_lines = [
        line for line in normalized_newlines.split("\n") if line.strip()
    ]
    # 模型完全无输出的情况（极少见，但需防御性处理）
    if not nonempty_lines:
        return None, False, "empty"

    # 第三步：单行输出时，尝试匹配 "Answer:" / "Final Answer:" 等标签前缀
    if len(nonempty_lines) == 1:
        line = nonempty_lines[0].strip()
        # ANSWER_LABEL_RE 匹配 "answer:" / "the answer is:" 等不区分大小写的标签
        label_match = ANSWER_LABEL_RE.fullmatch(line)
        # 若匹配标签，提取标签后的内容作为候选；否则整行作为候选
        candidate = label_match.group(1) if label_match else line
        candidate = candidate.strip()
        return candidate or None, bool(candidate), (
            "labeled" if label_match else "plain"
        )

    # 第四步：多行输出（含"思考过程" + "答案"的典型场景）
    # 扫描所有行，收集所有匹配标签模式的行
    labeled_candidates = []
    for index, line in enumerate(nonempty_lines):
        label_match = ANSWER_LABEL_RE.fullmatch(line.strip())
        if label_match:
            labeled_candidates.append((index, label_match.group(1).strip()))
    # 恰好一行带标签：提取该行标签后的内容，忽略其余行（视为思考过程）
    if len(labeled_candidates) == 1:
        _, candidate = labeled_candidates[0]
        return candidate or None, False, "labeled_with_extra_text"

    # 无标签或多标签的歧义情况：回退到取首行作为候选答案
    return nonempty_lines[0].strip() or None, False, "multiline"


def _parse_decimal(value: str) -> Decimal | None:
    """将 *value* 解析为 ``Decimal``，失败时返回 ``None``。"""
    if not DECIMAL_RE.fullmatch(value):
        return None
    try:
        return Decimal(value)
    except InvalidOperation:
        return None


def _base_result(
    raw_prediction: str,
    parsed_answer: str | None,
    parse_success: bool,
    format_valid: bool,
    extraction_method: str,
    parse_error: str | None,
) -> dict[str, Any]:
    """返回一个已填充公共字段的结果骨架。"""
    return {
        "prediction_raw": raw_prediction,
        "parsed_answer": parsed_answer,
        "parse_success": parse_success,
        "format_valid": format_valid,
        "extraction_method": extraction_method,
        "parse_error": parse_error,
        "strict_correct": False,
        "normalized_correct": False,
        "primary_correct": False,
    }


def evaluate_answer(
    task_type: str,
    gold_raw: str,
    prediction_raw: str,
) -> dict[str, Any]:
    """按任务特定规则解析并比较单条预测。

    返回包含 ``parse_success``、``format_valid``、``strict_correct``、
    ``normalized_correct`` 和 ``primary_correct`` 等字段的字典
    （详见 ``metrics.py`` 中的指标定义）。
    """
    if task_type not in TASK_ORDER:
        raise ValueError(f"unsupported task type: {task_type}")

    # 从原始生成文本中提取候选答案字符串
    candidate, clean_output, extraction_method = _extract_candidate(
        prediction_raw
    )
    # 无法提取任何候选答案 → 直接标记为解析失败
    if candidate is None:
        result = _base_result(
            prediction_raw,
            None,
            False,
            False,
            extraction_method,
            f"invalid {task_type.replace('_', ' ')} answer",
        )
        return result

    parse_success = False
    strict_format = False
    normalized_correct = False
    parse_error = None

    # ---- 按 task_type 选择对应的解析和比较策略 ----

    if task_type == "bit_manipulation":
        # 位操作题：答案必须为合法的二进制字符串
        parse_success = BINARY_ANSWER_RE.fullmatch(candidate) is not None
        strict_format = parse_success
        normalized_correct = parse_success and candidate == gold_raw
    elif task_type == "gravity":
        # 重力计算题：提取数值后用 Decimal 比较，避免浮点精度误差
        predicted_decimal = _parse_decimal(candidate)
        gold_decimal = _parse_decimal(gold_raw)
        if gold_decimal is None:
            raise ValueError(f"invalid gold gravity answer: {gold_raw!r}")
        parse_success = predicted_decimal is not None
        # 严格格式还要求候选满足完整浮点正则（如 0.51 而非 .51）
        strict_format = GRAVITY_ANSWER_RE.fullmatch(candidate) is not None
        normalized_correct = (
            parse_success and predicted_decimal == gold_decimal
        )
    elif task_type == "unit_conversion":
        # 单位换算题：与 gravity 逻辑相同，提取数值后用 Decimal 比较
        predicted_decimal = _parse_decimal(candidate)
        gold_decimal = _parse_decimal(gold_raw)
        if gold_decimal is None:
            raise ValueError(
                f"invalid gold unit conversion answer: {gold_raw!r}"
            )
        parse_success = predicted_decimal is not None
        strict_format = UNIT_ANSWER_RE.fullmatch(candidate) is not None
        normalized_correct = (
            parse_success and predicted_decimal == gold_decimal
        )
    elif task_type == "numeral":
        # 罗马数字题：大小写不敏感比较，strict_format 要求大写格式
        upper_candidate = candidate.upper()
        parse_success = (
            bool(upper_candidate)
            and ROMAN_RE.fullmatch(upper_candidate) is not None
        )
        strict_format = (
            parse_success and ROMAN_RE.fullmatch(candidate) is not None
        )
        normalized_correct = (
            parse_success and upper_candidate == gold_raw.upper()
        )
    elif task_type == "cipher":
        # 密码题：忽略空白和大小写后进行单词级比较
        # 将连续空白统一为单个空格，避免格式差异导致误判
        normalized_candidate = " ".join(candidate.lower().split())
        parse_success = (
            WORDS_RE.fullmatch(normalized_candidate) is not None
        )
        strict_format = WORDS_RE.fullmatch(candidate) is not None
        normalized_gold = " ".join(gold_raw.lower().split())
        normalized_correct = (
            parse_success and normalized_candidate == normalized_gold
        )
    else:
        # 符号/骰子题：答案为 1-4 个符号 token，需精确匹配
        parse_success = (
            1 <= len(candidate) <= 4
            and SYMBOL_TOKEN_RE.fullmatch(candidate) is not None
        )
        strict_format = parse_success
        normalized_correct = parse_success and candidate == gold_raw

    if not parse_success:
        parse_error = f"invalid {task_type.replace('_', ' ')} answer"

    # format_valid：clean_output（无多余文本）+ strict_format（格式校验）同时满足
    format_valid = clean_output and strict_format
    # strict_correct：解析成功 + 格式正确 + 与标准答案逐字符一致
    strict_correct = (
        parse_success
        and format_valid
        and candidate == gold_raw
    )
    # primary_correct：主流正确性指标
    # gravity 使用 normalized_correct（数值相等即可），因为 Decimal 输出格式多样
    # 其他 task_type 使用 strict_correct（要求精确匹配）
    if task_type == "gravity":
        primary_correct = format_valid and normalized_correct
    else:
        primary_correct = strict_correct

    result = _base_result(
        prediction_raw,
        candidate,
        parse_success,
        format_valid,
        extraction_method,
        parse_error,
    )
    result.update(
        {
            "strict_correct": strict_correct,
            "normalized_correct": normalized_correct,
            "primary_correct": primary_correct,
        }
    )
    return result
