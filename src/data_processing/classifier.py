"""按 prompt 结构对推理任务进行分类，并校验答案格式。

每种任务类型有固定的前缀行、预期的示例/查询模式和规范的答案格式。
当所有结构检查通过时 ``classify_record`` 返回任务类型，否则返回
``"unknown"`` 并附带失败原因。

| 任务               | 示例数 | 答案格式              |
|--------------------|--------|----------------------|
| bit_manipulation   | 7-10   | ``[01]{8}``         |
| gravity            | 3-5    | ``-?\d+\.\d{1,2}``  |
| unit_conversion    | 3-5    | ``-?\d+\.\d{2}``    |
| numeral            | 3-5    | 罗马数字             |
| cipher             | 3-5    | 小写英文单词         |
| symbolic_transform | 3-5    | 1-4 个符号字符       |
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence


TASK_ORDER = (
    "bit_manipulation",
    "gravity",
    "unit_conversion",
    "numeral",
    "cipher",
    "symbolic_transform",
)

# 每种任务类型的 prompt 第一行固定前缀文本。
# 分类时通过精确字符串比较第一行来识别任务类型，避免 regex 误判。
TASK_PREFIXES = {
    # 二进制位操作：8 位二进制数经过位移、旋转、XOR/AND/OR 等操作变换
    "bit_manipulation": (
        "In Alice's Wonderland, a secret bit manipulation rule transforms "
        "8-bit binary numbers. The transformation involves operations like "
        "bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority or "
        "choice functions."
    ),
    # 重力计算：重力常数 g 被秘密修改，需要根据自由落体公式 d = 0.5*g*t² 计算距离
    "gravity": (
        "In Alice's Wonderland, the gravitational constant has been secretly "
        "changed. Here are some example observations:"
    ),
    # 单位换算：长度（米）经过一个秘密换算因子转换成另一个数值
    "unit_conversion": (
        "In Alice's Wonderland, a secret unit conversion is applied to "
        "measurements. For example:"
    ),
    # 数字系统变换：阿拉伯数字被转换成一个不同的数字系统（实际为罗马数字）
    "numeral": (
        "In Alice's Wonderland, numbers are secretly converted into a "
        "different numeral system. Some examples are given below:"
    ),
    # 密码解密：小写英文单词经过加密规则变换，需要根据示例反推解密规则
    "cipher": (
        "In Alice's Wonderland, secret encryption rules are used on text. "
        "Here are some examples:"
    ),
    # 符号变换：数学符号表达式经过一组变换规则映射为另一个符号表达式
    "symbolic_transform": (
        "In Alice's Wonderland, a secret set of transformation rules is "
        "applied to equations. Below are a few examples:"
    ),
}

# 通用数字模式：支持可选的负号和可选的小数部分
NUMBER_PATTERN = r"-?\d+(?:\.\d+)?"

# --- bit_manipulation 的 regex ---
# 示例行格式："01001110 -> 10011100"，两个 8 位二进制串用 " -> " 分隔
BINARY_EXAMPLE_RE = re.compile(r"[01]{8} -> [01]{8}")
# 查询行："Now, determine the output for: <8位二进制串>"
BINARY_QUERY_RE = re.compile(r"Now, determine the output for: [01]{8}")
# 答案必须是精确的 8 位 0/1 字符串
BINARY_ANSWER_RE = re.compile(r"[01]{8}")

# --- gravity 的 regex ---
# 示例行格式："For t = 1.5s, distance = 11.03 m"
# NUMBER_PATTERN 用 rf-string 内联，同时匹配整数和小数形式的时间/距离
GRAVITY_EXAMPLE_RE = re.compile(
    rf"For t = {NUMBER_PATTERN}s, distance = {NUMBER_PATTERN} m"
)
# 查询行：固定公式 d = 0.5*g*t²，其中 g 是隐含的秘密重力常数
GRAVITY_QUERY_RE = re.compile(
    rf"Now, determine the falling distance for t = {NUMBER_PATTERN}s "
    r"given d = 0\.5\*g\*t\^2\."
)
# 答案：带符号的浮点数，1-2 位小数精度
GRAVITY_ANSWER_RE = re.compile(r"-?\d+\.\d{1,2}")

# --- unit_conversion 的 regex ---
# 示例行格式："5 m becomes 16.40"，输入值可能为整数
UNIT_EXAMPLE_RE = re.compile(rf"{NUMBER_PATTERN} m becomes {NUMBER_PATTERN}")
# 查询行："Now, convert the following measurement: <数字> m"
UNIT_QUERY_RE = re.compile(
    rf"Now, convert the following measurement: {NUMBER_PATTERN} m"
)
# 答案：带符号的浮点数，固定 2 位小数
UNIT_ANSWER_RE = re.compile(r"-?\d+\.\d{2}")

# --- numeral 的 regex ---
# 示例行格式："42 -> XLII"，阿拉伯数字映射到罗马数字
NUMERAL_EXAMPLE_RE = re.compile(r"\d+ -> [IVXLCDM]+")
# 查询行：要求将某个数字转换为"Wonderland 数字系统"
NUMERAL_QUERY_RE = re.compile(
    r"Now, write the number \d+ in the Wonderland numeral system\."
)
# 标准罗马数字正则：M(0-3个) + 百位(CM/CD/DC...) + 十位(XC/XL/LX...) + 个位(IX/IV/VI...)
# 涵盖了 1-3999 的所有有效罗马数字表示
ROMAN_RE = re.compile(
    r"M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})"
    r"(?:IX|IV|V?I{0,3})"
)

# --- cipher 的 regex ---
# 小写英文单词序列（空格分隔），用于匹配密码解密任务的答案
WORDS_RE = re.compile(r"[a-z]+(?: [a-z]+)*")
# 示例行格式："hello world -> ifmmp xpsme"，两端都是小写单词序列
CIPHER_EXAMPLE_RE = re.compile(
    r"[a-z]+(?: [a-z]+)* -> [a-z]+(?: [a-z]+)*"
)
# 查询行："Now, decrypt the following text: <加密后的小写文本>"
CIPHER_QUERY_RE = re.compile(
    r"Now, decrypt the following text: [a-z]+(?: [a-z]+)*"
)

# --- symbolic_transform 的 regex ---
# 符号 token：由一个或多个 ASCII 打印字符（不含字母）组成
# 字符类覆盖了数字、标点符号、运算符等，用于匹配如 "+-*/"、">>=" 等符号序列
SYMBOL_TOKEN_RE = re.compile(r"[0-9!-/:-@\[-`{-~]+")
# 查询行：使用了捕获组 (.+)，因为符号 token 的构成比较灵活，不能预定义
SYMBOL_QUERY_RE = re.compile(r"Now, determine the result for: (.+)")

# ID 格式：8 位十六进制小写字符串，用于校验 CSV 中的 id 字段
ID_RE = re.compile(r"[0-9a-f]{8}")


@dataclass(frozen=True)
class Classification:
    task_type: str
    reasons: tuple[str, ...] = ()


def _fullmatch(pattern: re.Pattern[str], value: str) -> bool:
    """``pattern.fullmatch(value)`` 的简写。"""
    return pattern.fullmatch(value) is not None


def _validate_example_block(
    lines: Sequence[str],
    example_pattern: re.Pattern[str],
    allowed_counts: set[int],
    header: str | None = None,
) -> list[str]:
    """校验示例区域的行数和每行格式。

    示例块的预期结构：
      行 0: 任务前缀（由上层校验）
      行 1: [可选的示例区域表头，如 "Here are some examples..."]
      行 2..N-1: 示例行，每行需匹配 example_pattern
      行 N: 空行或末尾

    header 参数为 None 表示该任务类型的示例块没有独立表头。
    """
    # 先过滤掉完全空的行，方便后续按位置索引
    content_lines = [line for line in lines if line]
    reasons: list[str] = []
    example_start = 1  # 默认示例从第 1 行开始（第 0 行是任务前缀）

    # 如果有表头，校验表头是否存在且内容匹配
    if header is not None:
        # 至少需要前缀行 + header 行，即 content_lines 长度 >= 2
        if len(content_lines) < 2 or content_lines[1] != header:
            reasons.append("invalid_example_header")
        else:
            example_start = 2  # 校验通过后示例从第 2 行开始

    # 剔除非示例内容：前缀行、表头行（如有）、最后一行（通常是"Now,..."查询）
    example_lines = content_lines[example_start:-1]

    # 数量校验：示例行数必须在允许范围内，否则说明生成逻辑有问题
    if len(example_lines) not in allowed_counts:
        reasons.append(f"unexpected_example_count:{len(example_lines)}")

    # 逐行校验格式：统计有多少行不匹配示例格式
    invalid_count = sum(
        not _fullmatch(example_pattern, line) for line in example_lines
    )
    if invalid_count:
        reasons.append(f"invalid_example_lines:{invalid_count}")
    return reasons


def _validate_symbolic_examples(lines: Sequence[str]) -> list[str]:
    """校验 ``symbolic_transform`` 的变长示例行。

    与其他任务不同，symbolic_transform 的示例格式为 "X = Y"，
    其中 X 和 Y 各是一个符号 token（如 "+-*/"），而非固定数字/字母格式。
    因此需要独立校验，不能复用 _validate_example_block。
    """
    content_lines = [line for line in lines if line]
    # symbolic_transform 没有示例表头行，示例从第 1 行开始直到倒数第 2 行
    example_lines = content_lines[1:-1]
    invalid_count = 0
    for line in example_lines:
        # 示例行必须包含 " = " 分隔符（注意两边有空格）
        # 用 " = " 而非 "=" 来精确匹配，避免把 ">=" 中的 "=" 误判为分隔符
        if " = " not in line:
            invalid_count += 1
            continue
        # 以第一个 " = " 为界，拆分为左侧输入和右侧输出
        left, right = line.split(" = ", 1)
        # 两端都必须是由符号字符构成的 token
        if not (
            _fullmatch(SYMBOL_TOKEN_RE, left)
            and _fullmatch(SYMBOL_TOKEN_RE, right)
        ):
            invalid_count += 1

    reasons = []
    # 允许 3-5 个示例（比 bit_manipulation 的 7-10 少，因为符号变换规则通常更简单）
    if len(example_lines) not in {3, 4, 5}:
        reasons.append(f"unexpected_example_count:{len(example_lines)}")
    if invalid_count:
        reasons.append(f"invalid_example_lines:{invalid_count}")
    return reasons


def _validate_query(
    lines: Sequence[str], query_pattern: re.Pattern[str]
) -> list[str]:
    """校验恰好有一条查询行匹配预期模式。

    查询行都以 "Now," 开头，在示例块之后、prompt 末尾的位置。
    必须恰好有一条，多或少数明 prompt 生成逻辑有 bug。
    """
    # 以 "Now," 开头的行就是查询行（所有任务类型的查询行都遵循这个约定）
    query_lines = [line for line in lines if line.startswith("Now,")]
    # 数量必须恰好为 1，0 条意味着缺少查询，>1 条意味着重复查询
    if len(query_lines) != 1:
        return [f"unexpected_query_count:{len(query_lines)}"]
    # 唯一的查询行必须完整匹配对应的查询正则模式
    if not _fullmatch(query_pattern, query_lines[0]):
        return ["invalid_query_format"]
    return []


def _is_canonical_roman(value: str) -> bool:
    """字符串是非空合法罗马数字时返回 ``True``。"""
    return bool(value) and _fullmatch(ROMAN_RE, value)


def classify_record(prompt: str, answer: str) -> Classification:
    """根据固定前缀进行分类，再交叉校验结构和答案格式。

    所有校验通过时返回 ``Classification(task_type)``，
    否则返回 ``Classification("unknown", reasons)``。

    分类流程：
    1. 用第一行精确匹配 TASK_PREFIXES，确定候选任务类型。
    2. 对每种任务类型运行专用的结构和答案校验。
    3. 全部通过 → 返回任务类型；任一失败 → 返回 "unknown" + 失败原因列表。
    """
    lines = prompt.splitlines()
    # 空 prompt 无类别可言
    if not lines:
        return Classification("unknown", ("empty_prompt",))

    # 用第一行去 TASK_PREFIXES 中做精确匹配
    prefix_matches = [
        task_type
        for task_type, prefix in TASK_PREFIXES.items()
        if lines[0] == prefix
    ]
    # 必须精确匹配到唯一一个任务类型（0 个 = 未知前缀，>1 个 = 前缀冲突）
    if len(prefix_matches) != 1:
        return Classification(
            "unknown", (f"prefix_match_count:{len(prefix_matches)}",)
        )

    task_type = prefix_matches[0]
    reasons: list[str] = []

    # --- 各任务类型的专属校验 ---
    # 每种任务类型的校验结构一致：示例块校验 → 查询行校验 → 答案格式校验

    if task_type == "bit_manipulation":
        # 示例校验：7-10 个示例行，有 "Here are some examples..." 表头
        # 示例格式：8 位二进制 -> 8 位二进制
        reasons.extend(
            _validate_example_block(
                lines,
                BINARY_EXAMPLE_RE,
                {7, 8, 9, 10},
                header="Here are some examples of input -> output:",
            )
        )
        # 查询校验："Now, determine the output for: <8位二进制>"
        reasons.extend(_validate_query(lines, BINARY_QUERY_RE))
        # 答案校验：必须是一个 8 位 0/1 字符串
        if not _fullmatch(BINARY_ANSWER_RE, answer):
            reasons.append("invalid_answer_format")

    elif task_type == "gravity":
        # 示例校验：3-5 个示例行，无独立表头（前缀本身充当了表头角色）
        reasons.extend(
            _validate_example_block(lines, GRAVITY_EXAMPLE_RE, {3, 4, 5})
        )
        # 查询校验：含自由落体公式 d=0.5*g*t²
        reasons.extend(_validate_query(lines, GRAVITY_QUERY_RE))
        # 答案校验：带符号浮点数，1-2 位小数
        if not _fullmatch(GRAVITY_ANSWER_RE, answer):
            reasons.append("invalid_answer_format")

    elif task_type == "unit_conversion":
        # 示例校验："<数字> m becomes <数字>" 格式
        reasons.extend(
            _validate_example_block(lines, UNIT_EXAMPLE_RE, {3, 4, 5})
        )
        reasons.extend(_validate_query(lines, UNIT_QUERY_RE))
        # 答案校验：精确 2 位小数的浮点数
        if not _fullmatch(UNIT_ANSWER_RE, answer):
            reasons.append("invalid_answer_format")

    elif task_type == "numeral":
        # 示例校验："<阿拉伯数字> -> <罗马数字>"
        reasons.extend(
            _validate_example_block(lines, NUMERAL_EXAMPLE_RE, {3, 4, 5})
        )
        reasons.extend(_validate_query(lines, NUMERAL_QUERY_RE))
        # 答案校验：必须是合法的罗马数字（非空 + 标准格式）
        # 用 _is_canonical_roman 而非简单 regex，因为它还检查了非空
        if not _is_canonical_roman(answer):
            reasons.append("invalid_answer_format")

    elif task_type == "cipher":
        # 示例校验：小写单词序列映射到另一个小写单词序列
        reasons.extend(
            _validate_example_block(lines, CIPHER_EXAMPLE_RE, {3, 4, 5})
        )
        reasons.extend(_validate_query(lines, CIPHER_QUERY_RE))
        # 答案校验：解密结果必须是小写单词序列
        if not _fullmatch(WORDS_RE, answer):
            reasons.append("invalid_answer_format")

    elif task_type == "symbolic_transform":
        # 使用独立校验函数，因为示例格式是 "X = Y" 而非 "X -> Y"
        reasons.extend(_validate_symbolic_examples(lines))
        # 查询行使用带捕获组(.+)的模式，因为符号表达式的形式不固定
        reasons.extend(_validate_query(lines, SYMBOL_QUERY_RE))
        # 额外校验：查询行中的符号 token 本身也必须合法
        query_lines = [line for line in lines if line.startswith("Now,")]
        if query_lines:
            query_match = SYMBOL_QUERY_RE.fullmatch(query_lines[0])
            # query_match.group(1) 是 "Now, determine the result for: " 之后的内容
            # 这部分必须由合法的符号字符组成
            if query_match and not _fullmatch(
                SYMBOL_TOKEN_RE, query_match.group(1)
            ):
                reasons.append("invalid_query_symbol_format")
        # 答案校验：1-4 个符号 token，长度限制避免了过长的无效答案
        if not (1 <= len(answer) <= 4) or not _fullmatch(
            SYMBOL_TOKEN_RE, answer
        ):
            reasons.append("invalid_answer_format")

    # 如果有任一校验失败，返回 "unknown" 并列出所有原因（去重 + 排序保证稳定性）
    if reasons:
        return Classification("unknown", tuple(sorted(set(reasons))))
    return Classification(task_type)
