#!/usr/bin/env python3
"""Generate Stage 2 thinking-format warmup JSONL data.

Stage 2 目标：让模型学会 thinking 输出协议（`<think>...</think>` + final answer），
同时混入 no-think 样本（Stage 1.5 strict replay + No Robots open replay），
避免模型学成“所有问题都强制输出 <think>”。

本脚本只做数据构建与检查，不启动训练，不修改 train_sft / 模型加载 / LoRA 配置 /
chat template，不使用官方 Wonderland validation/test，也不全量使用 OpenThoughts。

输出：
    data/instruction/stage2_thinking/train.jsonl
    data/instruction/stage2_thinking/validation.jsonl
    data/instruction/stage2_thinking/protocol_test.jsonl

assistant content 中不手写 <|im_end|>（chat template 会自动添加）。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import re
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence


# ---------------------------------------------------------------------------
# Thinking protocol markers (Qwen3 标准 thinking 标签，纯 ASCII)
# ---------------------------------------------------------------------------
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

# ---------------------------------------------------------------------------
# 五大类配额
# ---------------------------------------------------------------------------
TRAIN_QUOTAS: dict[str, int] = {
    "cot_collection_short": 500,
    "openthoughts_short": 300,
    "programmatic_thinking": 400,
    "stage1_5_strict_replay": 500,
    "no_robots_open_replay": 300,
}
VALIDATION_QUOTAS: dict[str, int] = {
    "cot_collection_short": 50,
    "openthoughts_short": 30,
    "programmatic_thinking": 40,
    "stage1_5_strict_replay": 50,
    "no_robots_open_replay": 30,
}
PROTOCOL_TEST_QUOTAS: dict[str, int] = dict(VALIDATION_QUOTAS)

CATEGORY_ORDER = list(TRAIN_QUOTAS.keys())

# programmatic_thinking 子类配额（略微偏向 binary_operation / wonderland_like_binary）
PROGRAMMATIC_SUBTYPES_TRAIN: dict[str, int] = {
    "simple_math_thinking": 54,
    "comparison_thinking": 54,
    "yes_no_reasoning_thinking": 54,
    "binary_operation_thinking": 62,
    "json_final_thinking": 54,
    "classification_thinking": 54,
    "wonderland_like_binary_thinking": 68,
}
PROGRAMMATIC_SUBTYPES_VAL: dict[str, int] = {
    "simple_math_thinking": 5,
    "comparison_thinking": 5,
    "yes_no_reasoning_thinking": 6,
    "binary_operation_thinking": 6,
    "json_final_thinking": 6,
    "classification_thinking": 5,
    "wonderland_like_binary_thinking": 7,
}

THINKING_CATEGORIES = {
    "cot_collection_short",
    "openthoughts_short",
    "programmatic_thinking",
}
NOTHINK_CATEGORIES = {"stage1_5_strict_replay", "no_robots_open_replay"}

# Stage 1.5 strict replay：优先保留这些子类，少量保留 extraction/line_count
STRICT_REPLAY_PREFERRED = {
    "exact_output",
    "binary_only",
    "wonderland_like_binary",
    "json_only",
    "yes_no",
    "stop_behavior",
}
STRICT_REPLAY_ALLOWED_EXTRA = {"extraction", "line_count"}
STRICT_BAD_PREFIXES = (
    "sure",
    "here is",
    "the answer is",
    "i hope this helps",
    "```",
)

# No Robots open replay：优先保留开放式回答类别
OPEN_REPLAY_CATEGORIES = {
    "Summarize",
    "Rewrite",
    "Open QA",
    "Closed QA",
    "Generation",
    "Coding",
    "Brainstorm",
    "Chat",
}
CLASSIFICATION_LABELS = {"positive", "negative", "neutral"}

DEFAULT_STAGE1_5_TRAIN = Path("data/instruction/stage1_5/train.jsonl")
DEFAULT_STAGE1_5_VALIDATION = Path("data/instruction/stage1_5/validation.jsonl")
DEFAULT_STAGE1_TRAIN = Path("data/instruction/stage1/train.jsonl")
DEFAULT_STAGE1_VALIDATION = Path("data/instruction/stage1/validation.jsonl")

_TOKENIZER_MODEL_CANDIDATES = (
    "models/Qwen3-1.7B-Base",
    "models/Qwen3-4B-Base",
    "Qwen/Qwen3-1.7B-Base",
)


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------
def log(message: str) -> None:
    timestamp = datetime.now().astimezone().strftime("%H:%M:%S")
    print(f"[{timestamp}] [stage2] {message}", file=sys.stderr, flush=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} invalid JSON: {exc}") from exc
    return rows


def write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(tmp, path)


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def looks_clean_text(text: str) -> bool:
    if not text:
        return False
    if "\x00" in text or "\ufffd" in text:
        return False
    stripped = text.strip()
    if not stripped:
        return False
    if len(set(stripped)) < 8:
        return False
    ratio = sum(ch.isprintable() or ch.isspace() for ch in text) / len(text)
    return ratio >= 0.98


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def prompt_hash(user_content: str) -> str:
    normalized = normalize_whitespace(user_content)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def make_messages(user: str, assistant: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def get_user_and_assistant(messages: Any) -> tuple[str, str]:
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    user_messages = [
        m.get("content")
        for m in messages
        if isinstance(m, Mapping) and m.get("role") == "user"
    ]
    assistant_messages = [
        m.get("content")
        for m in messages
        if isinstance(m, Mapping) and m.get("role") == "assistant"
    ]
    if not user_messages or not assistant_messages:
        raise ValueError("messages must contain user and assistant messages")
    user = user_messages[0]
    assistant = assistant_messages[-1]
    if not isinstance(user, str) or not isinstance(assistant, str):
        raise ValueError("message content must be strings")
    return user, assistant


# ---------------------------------------------------------------------------
# Token 估算（优先用本地 Qwen3 tokenizer，失败则用启发式回退）
# ---------------------------------------------------------------------------
class TokenEstimator:
    def __init__(self, cache_dir: Path | None = None) -> None:
        self._tokenizer: Any = None
        self._tried = False
        self._cache_dir = cache_dir

    def _load(self) -> Any | None:
        if self._tried:
            return self._tokenizer
        self._tried = True
        try:
            from transformers import AutoTokenizer

            for candidate in _TOKENIZER_MODEL_CANDIDATES:
                try:
                    tok = AutoTokenizer.from_pretrained(
                        candidate,
                        cache_dir=str(self._cache_dir) if self._cache_dir else None,
                        local_files_only=candidate.startswith("models/"),
                    )
                    self._tokenizer = tok
                    log(f"Token estimator: loaded tokenizer from {candidate}")
                    return tok
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
            log(
                "Token estimator: tokenizer unavailable, using heuristic fallback "
                f"(last error: {last_exc if 'last_exc' in locals() else 'none'})"
            )
        except Exception as exc:  # noqa: BLE001
            log(f"Token estimator: transformers import failed, heuristic fallback ({exc})")
        return self._tokenizer

    def count(self, text: str) -> int:
        if not text:
            return 0
        tok = self._load()
        if tok is not None:
            try:
                return len(tok(text, add_special_tokens=False)["input_ids"])
            except Exception:  # noqa: BLE001
                pass
        # 启发式回退：英文约 4 char/token，混合中英文按词数 * 1.3
        return max(1, int(len(text) / 3.5))


# ---------------------------------------------------------------------------
# thinking content 构造
# ---------------------------------------------------------------------------
def build_thinking_assistant(reasoning: str, final_answer: str) -> str:
    """构造 `<think>\n{reasoning}\n</think>\n\n{final_answer}`。"""
    reasoning_clean = reasoning.strip()
    final_clean = final_answer.strip()
    return f"{THINK_OPEN}\n{reasoning_clean}\n{THINK_CLOSE}\n\n{final_clean}"


def parse_thinking(content: str) -> tuple[str, str] | None:
    """从 assistant content 中抽取 (reasoning, final_answer)。

    要求 <think> 出现在 </think> 之前，且 </think> 后有非空 final answer。
    """
    open_idx = content.find(THINK_OPEN)
    close_idx = content.find(THINK_CLOSE)
    if open_idx == -1 or close_idx == -1 or open_idx >= close_idx:
        return None
    reasoning = content[open_idx + len(THINK_OPEN) : close_idx].strip()
    final = content[close_idx + len(THINK_CLOSE) :].strip()
    if not reasoning:
        return None
    return reasoning, final


# ---------------------------------------------------------------------------
# 外部数据适配层（CoT-Collection / OpenThoughts）
# ---------------------------------------------------------------------------
QUESTION_FIELDS = ("question", "input", "instruction", "prompt", "problem", "query", "source")
REASONING_FIELDS = (
    "rationale",
    "reasoning",
    "cot",
    "explanation",
    "thoughts",
    "thinking",
    "deep_thinking",
    "solution",
    "response",
)
ANSWER_FIELDS = (
    "answer",
    "output",
    "target",
    "final_answer",
    "final",
    "ground_truth",
    "gold",
)


def _record_fields(record: Mapping[str, Any]) -> list[str]:
    return [k for k, v in record.items() if not isinstance(v, (dict, list)) or k == "messages"]


def _extract_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and value:
        # list of strings or list of dicts with content
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                if isinstance(item.get("content"), str):
                    parts.append(item["content"])
                elif isinstance(item.get("text"), str):
                    parts.append(item["text"])
        if parts:
            return "\n".join(parts)
    if isinstance(value, Mapping):
        for key in ANSWER_FIELDS + REASONING_FIELDS + QUESTION_FIELDS:
            if isinstance(value.get(key), str):
                return value[key]
    return None


def _probe_field(record: Mapping[str, Any], candidates: Sequence[str]) -> str | None:
    for key in candidates:
        if key in record:
            text = _extract_text(record[key])
            if text and text.strip():
                return text
    return None


def print_field_probe(name: str, sample_records: Sequence[Mapping[str, Any]]) -> None:
    log(f"=== {name} 字段探测 ===")
    if not sample_records:
        log(f"{name}: no records to probe")
        return
    fields = sorted({k for r in sample_records for k in r.keys()})
    log(f"{name} detected fields: {fields}")
    for i, record in enumerate(sample_records[:3], start=1):
        summary: dict[str, str] = {}
        for key, value in record.items():
            text = _extract_text(value)
            if text is None:
                summary[key] = f"<{type(value).__name__}>"
            else:
                summary[key] = text[:160].replace("\n", " ")
        log(f"{name} sample {i}: {json.dumps(summary, ensure_ascii=False)}")


class ExternalAdapter:
    """通用外部 thinking 数据适配器：探测字段 -> 抽取 (question, reasoning, answer)。"""

    def __init__(
        self,
        name: str,
        records: Sequence[Mapping[str, Any]],
        source_tag: str,
    ) -> None:
        self.name = name
        self.records = list(records)
        self.source_tag = source_tag

    def adapt(
        self,
        max_reasoning_tokens: int,
        max_answer_tokens: int,
        max_total_tokens: int,
        estimator: TokenEstimator,
    ) -> list[dict[str, Any]]:
        """返回适配后的候选记录列表（未按 split 划分）。"""
        print_field_probe(self.name, self.records[:5])
        candidates: list[dict[str, Any]] = []
        stats = Counter()
        seen_question_hashes: set[str] = set()
        for index, record in enumerate(self.records):
            stats["total"] += 1
            question = _probe_field(record, QUESTION_FIELDS)
            reasoning = _probe_field(record, REASONING_FIELDS)
            answer = _probe_field(record, ANSWER_FIELDS)
            if not question:
                stats["missing_question"] += 1
                continue
            if not reasoning:
                stats["missing_reasoning"] += 1
                continue
            if not answer:
                stats["missing_answer"] += 1
                continue
            question = question.strip()
            reasoning = reasoning.strip()
            answer = answer.strip()
            if not (question and reasoning and answer):
                stats["empty_after_strip"] += 1
                continue
            # 源内 prompt-hash 去重（CoT-Collection 等存在大量模板重复）
            qhash = prompt_hash(question)
            if qhash in seen_question_hashes:
                stats["duplicate_question"] += 1
                continue
            seen_question_hashes.add(qhash)
            stats["field_ok"] += 1
            candidates.append(
                {
                    "question": question,
                    "reasoning": reasoning,
                    "answer": answer,
                    "original_index": index,
                    "original_id": str(record.get("id") or record.get("prompt_id") or record.get("_id") or ""),
                    "fields_used": {
                        "question": self._used_field(record, QUESTION_FIELDS),
                        "reasoning": self._used_field(record, REASONING_FIELDS),
                        "answer": self._used_field(record, ANSWER_FIELDS),
                    },
                }
            )
        log(f"{self.name} field probe stats: {dict(stats)}")
        return self._filter(candidates, max_reasoning_tokens, max_answer_tokens, max_total_tokens, estimator)

    @staticmethod
    def _used_field(record: Mapping[str, Any], candidates: Sequence[str]) -> str:
        for key in candidates:
            if key in record and _extract_text(record[key]):
                return key
        return ""

    def _filter(
        self,
        candidates: Sequence[Mapping[str, Any]],
        max_reasoning_tokens: int,
        max_answer_tokens: int,
        max_total_tokens: int,
        estimator: TokenEstimator,
    ) -> list[dict[str, Any]]:
        """子类覆盖：应用各自筛选规则。基类提供通用清洗过滤。"""
        raise NotImplementedError


class CotCollectionAdapter(ExternalAdapter):
    def _filter(self, candidates, max_reasoning_tokens, max_answer_tokens, max_total_tokens, estimator):
        out: list[dict[str, Any]] = []
        stats = Counter()
        for cand in candidates:
            stats["candidate"] += 1
            question = cand["question"]
            reasoning = cand["reasoning"]
            answer = cand["answer"]
            if not looks_clean_text(question) or not looks_clean_text(reasoning):
                stats["dirty_text"] += 1
                continue
            if "<" in reasoning and ">" in reasoning and re.search(r"<[a-zA-Z/][^>]*>", reasoning):
                stats["html_in_reasoning"] += 1
                continue
            r_tokens = estimator.count(reasoning)
            a_tokens = estimator.count(answer)
            q_tokens = estimator.count(question)
            total = r_tokens + a_tokens + q_tokens
            if r_tokens < 40 or r_tokens > 300:
                stats["reasoning_tokens_out_of_range"] += 1
                continue
            if a_tokens > 64:
                stats["answer_too_long"] += 1
                continue
            if total > max_total_tokens:
                stats["total_too_long"] += 1
                continue
            if len(question) > 1200:
                stats["prompt_too_long"] += 1
                continue
            if len(answer) > 400:
                stats["answer_chars_too_long"] += 1
                continue
            if self._looks_open_ended(question):
                stats["open_ended"] += 1
                continue
            stats["pass"] += 1
            cand["reasoning_tokens"] = r_tokens
            cand["answer_tokens"] = a_tokens
            cand["total_tokens"] = total
            out.append(cand)
        log(f"cot_collection filter stats: {dict(stats)}")
        return out

    @staticmethod
    def _looks_open_ended(question: str) -> bool:
        lowered = question.lower()
        markers = (
            "write an essay",
            "write a story",
            "write a poem",
            "in your own words, describe",
            "do you agree",
            "what is your opinion",
            "discuss your views",
        )
        return any(marker in lowered for marker in markers)


class OpenThoughtsAdapter(ExternalAdapter):
    """OpenThoughts (ShareGPT {conversations:[{from,value}], system}) 适配器。

    OpenThoughts-114k 是长 CoT 蒸馏数据：assistant 回复形如
    ``<|begin_of_thought|>\\n...长推理...\\n<|end_of_thought|>\\n\\n最终答案``。
    Stage 2 warmup 只需要"一点新式 reasoning 蒸馏风格"且要求短推理 + 短答案，
    因此本适配器会解析 thought 块并严格过滤，绝大多数长 CoT 样本会被拒绝。
    """

    OT_THOUGHT_OPEN = "<|begin_of_thought|>"
    OT_THOUGHT_CLOSE = "<|end_of_thought|>"

    def adapt(self, max_reasoning_tokens, max_answer_tokens, max_total_tokens, estimator):
        print_field_probe(self.name, self.records[:5])
        candidates: list[dict[str, Any]] = []
        stats = Counter()
        for index, record in enumerate(self.records):
            stats["total"] += 1
            parsed = self._parse_conversation(record)
            if parsed is None:
                stats["parse_fail"] += 1
                continue
            question, reasoning, answer = parsed
            stats["parsed"] += 1
            candidates.append(
                {
                    "question": question,
                    "reasoning": reasoning,
                    "answer": answer,
                    "original_index": index,
                    "original_id": str(record.get("id") or ""),
                    "fields_used": {"question": "conversations[user]", "reasoning": "begin_of_thought", "answer": "post_end_of_thought"},
                }
            )
        log(f"{self.name} conversation parse stats: {dict(stats)}")
        return self._filter(candidates, max_reasoning_tokens, max_answer_tokens, max_total_tokens, estimator)

    def _parse_conversation(self, record: Mapping[str, Any]) -> tuple[str, str, str] | None:
        convs = record.get("conversations")
        if not isinstance(convs, list) or not convs:
            return None
        user_msg = None
        assistant_msg = None
        for c in convs:
            if not isinstance(c, Mapping):
                continue
            role = c.get("from") or c.get("role")
            value = c.get("value") or c.get("content")
            if not isinstance(value, str):
                continue
            if role in ("user", "human"):
                if user_msg is None:
                    user_msg = value
            elif role in ("assistant", "gpt", "model"):
                assistant_msg = value
        if not user_msg or not assistant_msg:
            return None
        bo = self.OT_THOUGHT_OPEN
        bc = self.OT_THOUGHT_CLOSE
        if bo not in assistant_msg or bc not in assistant_msg:
            return None
        if assistant_msg.find(bo) >= assistant_msg.find(bc):
            return None
        reasoning = assistant_msg[assistant_msg.find(bo) + len(bo): assistant_msg.find(bc)].strip()
        answer = assistant_msg[assistant_msg.find(bc) + len(bc):].strip()
        if not reasoning or not answer:
            return None
        # final answer 可能仍含 markdown 包裹，剥离
        answer = self._strip_answer(answer)
        return user_msg, reasoning, answer

    @staticmethod
    def _strip_answer(answer: str) -> str:
        s = answer.strip()
        # 剥离首尾的代码块/粗体等包裹
        if s.startswith("```") and s.endswith("```"):
            inner = s[3:]
            if inner.startswith("\n"):
                inner = inner[1:]
            if inner.endswith("\n"):
                inner = inner[:-1]
            s = inner.strip()
        # 多段 answer：只取第一段非空行作为 final（Stage 2 要求短答案）
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        if len(lines) > 1:
            # 如果第一行很短像是 boxed answer，取它；否则取整段但截断
            first = lines[0]
            if len(first) <= 200:
                return first
        return s[:400]

    def _filter(self, candidates, max_reasoning_tokens, max_answer_tokens, max_total_tokens, estimator):
        out: list[dict[str, Any]] = []
        stats = Counter()
        # OpenThoughts 是长 CoT 蒸馏，标准 400/64 几乎过滤为空。
        # Stage 2 只取"一点新式 reasoning 蒸馏风格"，这里用稍宽松但仍可控的阈值：
        #   reasoning <= 800 tokens（远小于原始中位数 4051），
        #   answer <= 80 tokens（拒绝"一大段解释"的 final）。
        relaxed_r = min(max_reasoning_tokens * 2, 800)
        relaxed_a = min(max_answer_tokens * 2 + 16, 80)
        log(f"openthoughts relaxed filter: reasoning<={relaxed_r}t answer<={relaxed_a}t")
        for cand in candidates:
            stats["candidate"] += 1
            question = cand["question"]
            reasoning = cand["reasoning"]
            answer = cand["answer"]
            if not looks_clean_text(question) or not looks_clean_text(reasoning):
                stats["dirty_text"] += 1
                continue
            if "```" in reasoning:
                stats["code_in_reasoning"] += 1
                continue
            if question.count("def ") >= 1 or "executable Python" in question[:200]:
                stats["code_question"] += 1
                continue
            r_tokens = estimator.count(reasoning)
            a_tokens = estimator.count(answer)
            q_tokens = estimator.count(question)
            total = r_tokens + a_tokens + q_tokens
            if r_tokens > relaxed_r:
                stats["reasoning_too_long"] += 1
                continue
            if a_tokens > relaxed_a:
                stats["answer_too_long"] += 1
                continue
            if total > max_total_tokens:
                stats["total_too_long"] += 1
                continue
            if len(question) > 1200:
                stats["prompt_too_long"] += 1
                continue
            if len(answer) > 400:
                stats["answer_chars_too_long"] += 1
                continue
            stats["pass"] += 1
            cand["reasoning_tokens"] = r_tokens
            cand["answer_tokens"] = a_tokens
            cand["total_tokens"] = total
            out.append(cand)
        log(f"openthoughts filter stats: {dict(stats)}")
        return out


def _load_cot_collection_hub(
    target_count: int,
    dataset_id: str,
    max_reasoning_tokens: int,
    max_answer_tokens: int,
    max_total_tokens: int,
    estimator: "TokenEstimator",
) -> list[dict[str, Any]]:
    """流式加载并过滤 kaist-ai/CoT-Collection，返回适配后的候选记录。

    该数据集是一个 2.3GB 的单 JSON 字典文件（``data/CoT_collection_en.json``），
    顶层 ``{"id": {source, target, rationale, config, task, prompt}}``。
    数据集中前部有大量模板化/分类任务（同一 prompt 文本重复上千次），直接按
    ``limit`` 截取前 N 条会导致去重后几乎为空。因此这里用 ijson 流式扫描整个
    字典，在扫描过程中完成 prompt-hash 去重 + CoT 筛选，收集到 ``target_count``
    条即停止，避免把 2.3GB 全量加载到内存。

    返回的候选记录已是适配后格式（{question, reasoning, answer, ...}），
    跳过 CotCollectionAdapter 二次适配。
    """
    import os as _os

    proxy = _os.environ.get("HTTPS_PROXY") or _os.environ.get("HTTP_PROXY")
    log(f"Loading CoT-Collection JSON via ijson streaming (proxy={'on' if proxy else 'off'}, target={target_count})")
    try:
        from huggingface_hub import hf_hub_download
        import ijson
    except ImportError as exc:
        log(f"CoT-Collection load failed (missing dep): {exc}")
        return []
    try:
        json_path = hf_hub_download(dataset_id, "data/CoT_collection_en.json", repo_type="dataset")
    except Exception as exc:  # noqa: BLE001
        log(f"CoT-Collection download failed: {exc}")
        return []

    stats = Counter()
    seen_hashes: set[str] = set()
    candidates: list[dict[str, Any]] = []
    scanned = 0
    # 先打印前 3 条原始样本摘要（字段探测）
    probed = 0
    try:
        with open(json_path, "rb") as handle:
            for key, value in ijson.kvitems(handle, ""):
                scanned += 1
                if not isinstance(value, dict):
                    continue
                if probed < 3:
                    summary = {
                        "_id": str(key),
                        "source": str(value.get("source", ""))[:160].replace("\n", " "),
                        "target": str(value.get("target", ""))[:160].replace("\n", " "),
                        "rationale": str(value.get("rationale", ""))[:160].replace("\n", " "),
                        "task": str(value.get("task", "")),
                    }
                    log(f"cot_collection sample {probed + 1}: {json.dumps(summary, ensure_ascii=False)}")
                    probed += 1
                    if probed == 3:
                        log(f"cot_collection detected fields: {sorted(value.keys()) + ['_id']}")
                stats["total"] += 1
                question = value.get("source")
                reasoning = value.get("rationale")
                answer = value.get("target")
                if not (isinstance(question, str) and isinstance(reasoning, str) and isinstance(answer, str)):
                    stats["missing_field"] += 1
                    continue
                question, reasoning, answer = question.strip(), reasoning.strip(), answer.strip()
                if not (question and reasoning and answer):
                    stats["empty"] += 1
                    continue
                qh = prompt_hash(question)
                if qh in seen_hashes:
                    stats["duplicate"] += 1
                    continue
                seen_hashes.add(qh)
                # CoT 筛选规则
                if not looks_clean_text(question) or not looks_clean_text(reasoning):
                    stats["dirty"] += 1
                    continue
                if "<" in reasoning and ">" in reasoning and re.search(r"<[a-zA-Z/][^>]*>", reasoning):
                    stats["html"] += 1
                    continue
                r_tokens = estimator.count(reasoning)
                a_tokens = estimator.count(answer)
                q_tokens = estimator.count(question)
                total = r_tokens + a_tokens + q_tokens
                if r_tokens < 40 or r_tokens > max_reasoning_tokens:
                    stats["r_range"] += 1
                    continue
                if a_tokens > max_answer_tokens:
                    stats["a_long"] += 1
                    continue
                if total > max_total_tokens:
                    stats["total_long"] += 1
                    continue
                if len(question) > 1200:
                    stats["q_long"] += 1
                    continue
                if len(answer) > 400:
                    stats["a_chars"] += 1
                    continue
                if CotCollectionAdapter._looks_open_ended(question):
                    stats["open_ended"] += 1
                    continue
                stats["pass"] += 1
                candidates.append(
                    {
                        "question": question,
                        "reasoning": reasoning,
                        "answer": answer,
                        "original_index": scanned - 1,
                        "original_id": str(key),
                        "fields_used": {"question": "source", "reasoning": "rationale", "answer": "target"},
                        "reasoning_tokens": r_tokens,
                        "answer_tokens": a_tokens,
                        "total_tokens": total,
                    }
                )
                if len(candidates) >= target_count:
                    break
    except Exception as exc:  # noqa: BLE001
        log(f"CoT-Collection ijson parse failed: {exc}")
    log(f"CoT-Collection stream stats: scanned={scanned} {dict(stats)} kept={len(candidates)}")
    return candidates


def load_external_records(
    name: str,
    path: Path | None,
    dataset_id: str | None,
    split: str,
    limit: int | None,
) -> list[dict[str, Any]]:
    """加载外部数据：优先本地 JSONL，其次 datasets.load_dataset。

    特例：CoT-Collection（kaist-ai/CoT-Collection）是单 JSON 大字典文件，
    走专用 ijson 流式路径。
    """
    if path is not None:
        if not path.exists():
            raise ValueError(f"{name} local path does not exist: {path}")
        log(f"Loading {name} from local path: {path}")
        records = read_jsonl(path)
        if limit:
            records = records[:limit]
        log(f"{name} loaded {len(records)} records from local path")
        return records
    if dataset_id is None:
        return []
    # 注意：kaist-ai/CoT-Collection 的专用 ijson 流式路径在 build_candidate_pools
    # 中直接调用 _load_cot_collection_hub，不经过本函数（因为该数据集是单 2.3GB JSON
    # 字典而非 parquet split，需要边扫描边去重边筛选）。其他 dataset_id 走通用 streaming。
    log(f"Loading {name} from HuggingFace Hub: {dataset_id} (split={split}, streaming)")
    try:
        from datasets import load_dataset

        # 使用 streaming 模式，避免把超大数据集（如 OpenThoughts-114k）整包下载到本地。
        # 仅取前 `limit` 条记录即可满足 Stage 2 少量蒸馏需求。
        ds = load_dataset(dataset_id, split=split, streaming=True)
        records: list[dict[str, Any]] = []
        for record in ds:
            records.append(dict(record))
            if limit and len(records) >= limit:
                break
        log(f"{name} loaded {len(records)} records from Hub (streaming)")
        return records
    except Exception as exc:  # noqa: BLE001
        log(f"{name} Hub load failed: {exc}")
        return []


# ---------------------------------------------------------------------------
# 程序化 thinking 生成器（7 个子类，纯规则合成，不使用官方 Wonderland 数据）
# ---------------------------------------------------------------------------
def _eight_bit(rng: random.Random) -> str:
    return format(rng.randrange(256), "08b")


def _rotate_left(bits: str, amount: int) -> str:
    return bits[amount:] + bits[:amount]


def _rotate_right(bits: str, amount: int) -> str:
    return bits[-amount:] + bits[:-amount]


def _bitwise_op(bits: str, mask: str, op: Callable[[int, int], int]) -> str:
    return "".join(str(op(int(b), int(m))) for b, m in zip(bits, mask))


def _build_binary_rule(rng: random.Random) -> tuple[str, Callable[[str], str], str]:
    """返回 (rule_name, transform, human_description)。仅 toy 规则。"""
    rule_name = rng.choice(
        ["not", "rotate_left", "rotate_right", "xor", "and", "or", "shift_left", "shift_right"]
    )
    if rule_name == "not":
        return "bitwise_not", lambda b: "".join("1" if x == "0" else "0" for x in b), "flip each bit (bitwise NOT)"
    if rule_name == "rotate_left":
        amt = rng.randint(1, 3)
        return f"rotate_left_{amt}", lambda b: _rotate_left(b, amt), f"rotate left by {amt}"
    if rule_name == "rotate_right":
        amt = rng.randint(1, 3)
        return f"rotate_right_{amt}", lambda b: _rotate_right(b, amt), f"rotate right by {amt}"
    if rule_name == "xor":
        mask = _eight_bit(rng)
        return f"xor_{mask}", lambda b: _bitwise_op(b, mask, lambda a, c: a ^ c), f"XOR with {mask}"
    if rule_name == "and":
        mask = _eight_bit(rng)
        return f"and_{mask}", lambda b: _bitwise_op(b, mask, lambda a, c: a & c), f"AND with {mask}"
    if rule_name == "or":
        mask = _eight_bit(rng)
        return f"or_{mask}", lambda b: _bitwise_op(b, mask, lambda a, c: a | c), f"OR with {mask}"
    if rule_name == "shift_left":
        amt = rng.randint(1, 3)
        return f"shift_left_{amt}", lambda b: b[amt:] + ("0" * amt), f"shift left by {amt}"
    amt = rng.randint(1, 3)
    return f"shift_right_{amt}", lambda b: ("0" * amt) + b[:-amt], f"shift right by {amt}"


def _compact_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def gen_simple_math_thinking(rng: random.Random) -> dict[str, Any]:
    op = rng.choice(["add", "sub", "mul"])
    if op == "add":
        a, b = rng.randint(2, 99), rng.randint(2, 99)
        result = a + b
        question = f"What is {a} + {b}? Think briefly, then give only the final number after {THINK_CLOSE}."
        reasoning = f"{a} + {b} = {result}."
        final = str(result)
    elif op == "sub":
        a = rng.randint(20, 99)
        b = rng.randint(2, a)
        result = a - b
        question = f"What is {a} - {b}? Think briefly, then give only the final number after {THINK_CLOSE}."
        reasoning = f"{a} - {b} = {result}."
        final = str(result)
    else:
        a, b = rng.randint(2, 12), rng.randint(2, 12)
        result = a * b
        question = f"What is {a} * {b}? Think briefly, then give only the final number after {THINK_CLOSE}."
        reasoning = f"{a} * {b} = {result}."
        final = str(result)
    return {"user": question, "reasoning": reasoning, "final_answer": final}


def gen_comparison_thinking(rng: random.Random) -> dict[str, Any]:
    a, b = rng.randint(2, 999), rng.randint(2, 999)
    if a == b:
        b += 1
    if a > b:
        larger, reasoning = a, f"{a} > {b}, so {a} is greater."
        final = str(a)
    else:
        larger, reasoning = b, f"{b} > {a}, so {b} is greater."
        final = str(b)
    question = (
        f"Which is greater, {a} or {b}? Think briefly, then give only the larger number after {THINK_CLOSE}."
    )
    return {"user": question, "reasoning": reasoning, "final_answer": final}


def gen_yes_no_reasoning_thinking(rng: random.Random) -> dict[str, Any]:
    """Multi-step yes/no reasoning: compute first, then check condition."""
    kind = rng.choice(["greater", "even", "divisible", "equal"])
    if kind == "greater":
        a, b, c = rng.randint(10, 99), rng.randint(10, 99), rng.randint(50, 150)
        total = a + b
        answer = "yes" if total > c else "no"
        reasoning = f"First, {a} + {b} = {total}. Then compare: {total} > {c} is {answer}."
        question = f"Is {a} + {b} greater than {c}? Think briefly, then answer only yes or no after {THINK_CLOSE}."
    elif kind == "even":
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        total = a + b
        answer = "yes" if total % 2 == 0 else "no"
        reasoning = f"First, {a} + {b} = {total}. {total} is {'even' if answer == 'yes' else 'odd'}, so the answer is {answer}."
        question = f"Is {a} + {b} an even number? Think briefly, then answer only yes or no after {THINK_CLOSE}."
    elif kind == "divisible":
        a, b = rng.randint(5, 20), rng.randint(5, 20)
        product = a * b
        d = rng.choice([2, 3, 4, 5, 6, 7, 8, 9])
        remainder = product % d
        answer = "yes" if remainder == 0 else "no"
        reasoning = f"First, {a} x {b} = {product}. Then {product} / {d} = {product // d} remainder {remainder}, so {product} is {'divisible' if answer == 'yes' else 'not divisible'} by {d}."
        question = f"Is {a} x {b} divisible by {d}? Think briefly, then answer only yes or no after {THINK_CLOSE}."
    else:
        a, b = rng.randint(2, 25), rng.randint(2, 25)
        product = a * b
        c = product if rng.random() < 0.4 else rng.randint(product - 30, product + 30)
        answer = "yes" if product == c else "no"
        reasoning = f"First, {a} x {b} = {product}. Then compare: {product} == {c} is {answer}."
        question = f"Is {a} x {b} equal to {c}? Think briefly, then answer only yes or no after {THINK_CLOSE}."
    return {"user": question, "reasoning": reasoning, "final_answer": answer}


def gen_binary_operation_thinking(rng: random.Random) -> dict[str, Any]:
    rule_name, transform, desc = _build_binary_rule(rng)
    bits = _eight_bit(rng)
    result = transform(bits)
    if rule_name == "bitwise_not":
        question = (
            f"Apply bitwise NOT to {bits}. Think briefly, then output only the 8-bit result after {THINK_CLOSE}."
        )
        reasoning = f"Flipping each bit of {bits} gives {result}."
    elif rule_name.startswith("rotate_left"):
        question = (
            f"Rotate {bits} left by {rule_name.split('_')[-1]} positions. "
            f"Think briefly, then output only the 8-bit result after {THINK_CLOSE}."
        )
        reasoning = f"Rotating {bits} left by {rule_name.split('_')[-1]} gives {result}."
    elif rule_name.startswith("rotate_right"):
        question = (
            f"Rotate {bits} right by {rule_name.split('_')[-1]} positions. "
            f"Think briefly, then output only the 8-bit result after {THINK_CLOSE}."
        )
        reasoning = f"Rotating {bits} right by {rule_name.split('_')[-1]} gives {result}."
    elif rule_name.startswith("xor"):
        mask = rule_name.split("_", 1)[1]
        question = (
            f"Compute {bits} XOR {mask}. Think briefly, then output only the 8-bit result after {THINK_CLOSE}."
        )
        reasoning = f"Bitwise XOR of {bits} with {mask} gives {result}."
    elif rule_name.startswith("and"):
        mask = rule_name.split("_", 1)[1]
        question = (
            f"Compute {bits} AND {mask}. Think briefly, then output only the 8-bit result after {THINK_CLOSE}."
        )
        reasoning = f"Bitwise AND of {bits} with {mask} gives {result}."
    elif rule_name.startswith("or"):
        mask = rule_name.split("_", 1)[1]
        question = (
            f"Compute {bits} OR {mask}. Think briefly, then output only the 8-bit result after {THINK_CLOSE}."
        )
        reasoning = f"Bitwise OR of {bits} with {mask} gives {result}."
    elif rule_name.startswith("shift_left"):
        amt = rule_name.split("_")[-1]
        question = (
            f"Shift {bits} left by {amt} (zero-fill). Think briefly, then output only the 8-bit result after {THINK_CLOSE}."
        )
        reasoning = f"Shifting {bits} left by {amt} gives {result}."
    else:
        amt = rule_name.split("_")[-1]
        question = (
            f"Shift {bits} right by {amt} (zero-fill). Think briefly, then output only the 8-bit result after {THINK_CLOSE}."
        )
        reasoning = f"Shifting {bits} right by {amt} gives {result}."
    return {"user": question, "reasoning": reasoning, "final_answer": result, "rule": rule_name}


def gen_json_final_thinking(rng: random.Random) -> dict[str, Any]:
    kind = rng.choice(["sum", "product", "answer", "label", "binary"])
    if kind == "sum":
        a, b = rng.randint(0, 50), rng.randint(0, 50)
        value = a + b
        reasoning = f"{a} + {b} = {value}."
        final = _compact_json({"sum": value})
        question = (
            f"What is {a} + {b}? Think briefly, then output only the final JSON "
            f'(key "sum") after {THINK_CLOSE}.'
        )
    elif kind == "product":
        a, b = rng.randint(0, 12), rng.randint(0, 12)
        value = a * b
        reasoning = f"{a} * {b} = {value}."
        final = _compact_json({"product": value})
        question = (
            f"What is {a} * {b}? Think briefly, then output only the final JSON "
            f'(key "product") after {THINK_CLOSE}.'
        )
    elif kind == "answer":
        value = rng.choice(["yes", "no"])
        a, b = rng.randint(0, 50), rng.randint(0, 50)
        actual = "yes" if a > b else "no"
        reasoning = f"{a} > {b} is {actual}."
        final = _compact_json({"answer": actual})
        question = (
            f"Is {a} greater than {b}? Think briefly, then output only the final JSON "
            f'(key "answer") after {THINK_CLOSE}.'
        )
    elif kind == "label":
        text = rng.choice(
            ["The new feature works great and I love it.", "The app crashes every time I open it."]
        )
        label = "positive" if "love" in text else "negative"
        reasoning = f"The text expresses a {'positive' if label == 'positive' else 'negative'} sentiment."
        final = _compact_json({"label": label})
        question = (
            f'Classify the sentiment of: "{text}" Think briefly, then output only the final JSON '
            f'(key "label") after {THINK_CLOSE}.'
        )
    else:
        bits = _eight_bit(rng)
        flipped = "".join("1" if x == "0" else "0" for x in bits)
        reasoning = f"Flipping each bit of {bits} gives {flipped}."
        final = _compact_json({"binary": flipped})
        question = (
            f"Apply bitwise NOT to {bits}. Think briefly, then output only the final JSON "
            f'(key "binary") after {THINK_CLOSE}.'
        )
    return {"user": question, "reasoning": reasoning, "final_answer": final}


_SENTIMENT_EXAMPLES = {
    "positive": [
        "I love this product.",
        "The service was fast and friendly.",
        "This update works beautifully.",
        "I am happy with the result.",
        "The staff went above and beyond to help me.",
        "What a wonderful experience from start to finish.",
        "The new design looks amazing and loads quickly.",
        "I am thoroughly impressed with the build quality.",
        "Support resolved my issue in under five minutes.",
        "The food was delicious and the portions generous.",
        "This is the best purchase I have made all year.",
        "The team delivered ahead of schedule and exceeded expectations.",
        "I really enjoy using this app every single day.",
        "The packaging was neat and the item arrived intact.",
        "Excellent value for the price, highly recommended.",
        "The interface is intuitive and the performance is snappy.",
        "Their customer care is polite and genuinely helpful.",
        "I am satisfied with how smoothly everything went.",
    ],
    "negative": [
        "I hate how slow this app is.",
        "The package arrived broken.",
        "This was a frustrating experience.",
        "The result is disappointing.",
        "The device stopped working after only two days.",
        "Customer support kept me waiting for hours with no answer.",
        "The quality is much worse than the pictures suggested.",
        "I regret buying this; it does not do what it claims.",
        "The checkout process was confusing and buggy.",
        "My order was delayed by over a week with no updates.",
        "The battery drains far too quickly even when idle.",
        "The instructions were unclear and missing several steps.",
        "This update broke features that used to work fine.",
        "The material feels cheap and started fraying immediately.",
        "I would not recommend this to anyone at this price.",
        "The app crashes every time I try to open my profile.",
        "Shipping cost was hidden until the very last step.",
        "The whole experience left me feeling ignored as a customer.",
    ],
    "neutral": [
        "The meeting starts at noon.",
        "The box contains three cables.",
        "The report was published today.",
        "The device weighs two kilograms.",
        "The train arrives at platform four at 3 PM.",
        "The document has been forwarded to the legal team.",
        "The store is located on the second floor of the mall.",
        "The invoice number is 4821 and dated last Friday.",
        "The package includes a manual and a warranty card.",
        "The session is scheduled to last approximately one hour.",
        "The form must be submitted by the end of the week.",
        "The committee will review the proposal next Tuesday.",
        "The product is available in three different colors.",
        "The agreement was signed by both parties yesterday.",
        "The office is closed on public holidays.",
        "The system sends a confirmation email after each upload.",
        "The route covers approximately twenty kilometers.",
        "The announcement was made via the company newsletter.",
    ],
}


_CLASSIFICATION_TEMPLATES = [
    (
        "Classify the sentiment as positive, negative, or neutral. Think briefly, "
        "then output only the label after {close}.\nText: {text}"
    ),
    (
        "What is the sentiment of the following text: positive, negative, or neutral? "
        "Think briefly, then output only the label after {close}.\nText: {text}"
    ),
    (
        "Sentiment analysis. Choose one label: positive, negative, neutral. "
        "Think briefly, then output only the label after {close}.\nText: {text}"
    ),
    (
        "Read the text and classify its sentiment as positive, negative, or neutral. "
        "Think briefly, then output only the label after {close}.\nText: {text}"
    ),
]


def gen_classification_thinking(rng: random.Random) -> dict[str, Any]:
    label = rng.choice(sorted(_SENTIMENT_EXAMPLES))
    text = rng.choice(_SENTIMENT_EXAMPLES[label])
    template = rng.choice(_CLASSIFICATION_TEMPLATES)
    reasoning = f"The text expresses a {label} sentiment."
    question = template.format(close=THINK_CLOSE, text=text)
    return {"user": question, "reasoning": reasoning, "final_answer": label}


def gen_wonderland_like_binary_thinking(rng: random.Random) -> dict[str, Any]:
    rule_name, transform, desc = _build_binary_rule(rng)
    inputs = set()
    while len(inputs) < 5:
        inputs.add(_eight_bit(rng))
    values = list(inputs)
    examples = values[:4]
    query = values[4]
    answer = transform(query)
    example_lines = "\n".join(f"{bits} -> {transform(bits)}" for bits in examples)
    user = (
        "In Alice's Wonderland, a simple 8-bit transformation rule is shown by examples:\n"
        f"{example_lines}\n\n"
        f"Now determine the output for: {query}.\n"
        f"Think briefly, then output only the final 8-bit answer after {THINK_CLOSE}."
    )
    reasoning = f"The examples show the rule is {desc}. Applying it to {query} gives {answer}."
    return {"user": user, "reasoning": reasoning, "final_answer": answer, "rule": rule_name}


PROGRAMMATIC_GENERATORS: dict[str, Callable[[random.Random], dict[str, Any]]] = {
    "simple_math_thinking": gen_simple_math_thinking,
    "comparison_thinking": gen_comparison_thinking,
    "yes_no_reasoning_thinking": gen_yes_no_reasoning_thinking,
    "binary_operation_thinking": gen_binary_operation_thinking,
    "json_final_thinking": gen_json_final_thinking,
    "classification_thinking": gen_classification_thinking,
    "wonderland_like_binary_thinking": gen_wonderland_like_binary_thinking,
}


def generate_programmatic_candidates(
    split: str,
    subtype_counts: Mapping[str, int],
    rng: random.Random,
    buffer: int | None = None,
) -> list[dict[str, Any]]:
    """生成 programmatic thinking 候选记录，按子类配额平衡。

    每个子类按 prompt-hash 去重后保留至 ``quota + buffer`` 条，确保各子类分布
    接近配额（略微偏向 binary_operation / wonderland_like_binary），并留出 buffer
    供全局跨 split 去重时被丢弃后仍能满足配额。buffer 默认为 max(12, quota//4)，
    validation/protocol_test 使用 max(12, quota) 应对跨 split 去重。
    """
    candidates: list[dict[str, Any]] = []
    for subtype, count in subtype_counts.items():
        gen = PROGRAMMATIC_GENERATORS[subtype]
        if buffer is not None:
            sub_buffer = buffer
        elif split == "train":
            sub_buffer = max(12, count // 4)
        else:
            sub_buffer = max(12, count * 2)
        cap = count + sub_buffer
        seen: set[str] = set()
        produced = 0
        attempts = 0
        max_attempts = cap * 25
        while produced < cap and attempts < max_attempts:
            attempts += 1
            sample = gen(rng)
            h = prompt_hash(sample["user"])
            if h in seen:
                continue
            seen.add(h)
            candidates.append({**sample, "subtype": subtype})
            produced += 1
    rng.shuffle(candidates)
    return candidates


def programmatic_candidate_to_record(
    candidate: Mapping[str, Any],
    split: str,
    index: int,
    estimator: TokenEstimator,
) -> dict[str, Any]:
    subtype = candidate["subtype"]
    reasoning = candidate["reasoning"]
    final_answer = candidate["final_answer"]
    assistant = build_thinking_assistant(reasoning, final_answer)
    r_tokens = estimator.count(reasoning)
    total = estimator.count(candidate["user"]) + estimator.count(assistant)
    return {
        "id": f"stage2-{split}-programmatic_thinking-{subtype}-{index:05d}",
        "category": "programmatic_thinking",
        "source": "programmatic",
        "messages": make_messages(candidate["user"], assistant),
        "metadata": {
            "thinking": True,
            "final_answer": final_answer,
            "reasoning_token_estimate": r_tokens,
            "total_token_estimate": total,
            "source_original_id": "",
            "split": split,
            "subtype": subtype,
            "rule": candidate.get("rule", ""),
        },
    }


# ---------------------------------------------------------------------------
# 外部 thinking 候选 -> 记录
# ---------------------------------------------------------------------------
def external_candidate_to_record(
    candidate: Mapping[str, Any],
    split: str,
    category: str,
    source_tag: str,
    index: int,
    estimator: TokenEstimator,
) -> dict[str, Any]:
    question = candidate["question"]
    reasoning = normalize_whitespace(candidate["reasoning"])
    final_answer = normalize_whitespace(candidate["answer"])
    user = (
        f"Solve the problem. Think briefly, then give the final answer after {THINK_CLOSE}.\n\n"
        f"Problem: {question}"
    )
    assistant = build_thinking_assistant(reasoning, final_answer)
    r_tokens = estimator.count(reasoning)
    total = estimator.count(user) + estimator.count(assistant)
    return {
        "id": f"stage2-{split}-{category}-{index:05d}",
        "category": category,
        "source": source_tag,
        "messages": make_messages(user, assistant),
        "metadata": {
            "thinking": True,
            "final_answer": final_answer,
            "reasoning_token_estimate": r_tokens,
            "total_token_estimate": total,
            "source_original_id": candidate.get("original_id", ""),
            "split": split,
            "fields_used": candidate.get("fields_used", {}),
        },
    }


# ---------------------------------------------------------------------------
# Stage 1.5 strict replay（no-think，保持原 assistant 格式）
# ---------------------------------------------------------------------------
def stage1_5_strict_candidate_ok(row: Mapping[str, Any]) -> tuple[bool, str]:
    """校验单条 stage1_5 strict 记录是否可重放。返回 (ok, category)。"""
    category = row.get("category")
    if not isinstance(category, str):
        return False, ""
    if category == "no_robots_replay":
        return False, ""
    messages = row.get("messages")
    try:
        user, assistant = get_user_and_assistant(messages)
    except ValueError:
        return False, category
    if not user or not assistant:
        return False, category
    if "<|im_end|>" in assistant:
        return False, category
    stripped = assistant.strip()
    if assistant != stripped:
        return False, category
    lowered = stripped.lower()
    if any(lowered.startswith(prefix) for prefix in STRICT_BAD_PREFIXES):
        return False, category
    if "```" in stripped:
        return False, category
    if category in {"binary_only", "wonderland_like_binary"}:
        if not re.fullmatch(r"[01]{8}", stripped):
            return False, category
    elif category == "json_only":
        try:
            json.loads(stripped)
        except json.JSONDecodeError:
            return False, category
    elif category == "yes_no":
        if stripped not in {"yes", "no"}:
            return False, category
    elif category == "classification":
        if stripped not in CLASSIFICATION_LABELS:
            return False, category
    return True, category


def load_stage1_5_strict_candidates(path: Path) -> list[dict[str, Any]]:
    """加载并过滤 stage1_5 strict 候选，按优先级排序。"""
    rows = read_jsonl(path)
    preferred: list[dict[str, Any]] = []
    extra: list[dict[str, Any]] = []
    classification: list[dict[str, Any]] = []
    stats = Counter()
    for row in rows:
        ok, category = stage1_5_strict_candidate_ok(row)
        if not ok:
            stats["rejected"] += 1
            continue
        stats["ok"] += 1
        if category in STRICT_REPLAY_PREFERRED:
            preferred.append(row)
        elif category in STRICT_REPLAY_ALLOWED_EXTRA:
            extra.append(row)
        elif category == "classification":
            classification.append(row)
        else:
            stats["unknown_category"] += 1
    log(f"stage1_5 strict replay candidates from {path}: {dict(stats)}")
    return preferred + extra + classification


_STAGE1_5_MODULE: Any = None


def _load_stage1_5_module() -> Any:
    """延迟加载 scripts/generate_stage1_5_strict_data.py，复用其 strict 生成器。"""
    global _STAGE1_5_MODULE
    if _STAGE1_5_MODULE is not None:
        return _STAGE1_5_MODULE
    import importlib.util

    script_path = Path(__file__).resolve().parent / "generate_stage1_5_strict_data.py"
    if not script_path.exists():
        raise FileNotFoundError(f"stage1_5 generator script not found: {script_path}")
    spec = importlib.util.spec_from_file_location("stage1_5_strict_data", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    _STAGE1_5_MODULE = module
    return module


def generate_fresh_strict_candidates(
    count: int, rng: random.Random, source_tag: str = "stage1_5_fresh"
) -> list[dict[str, Any]]:
    """重新生成未见 strict 样本（no-think），用于补足 validation/protocol_test。

    复用 stage1_5 的 GENERATORS，按 prompt-hash 去重，返回 stage1_5 格式记录
    （含 messages/category/validator），后续由 strict_replay_to_record 转换。
    仅使用 STRICT_REPLAY_PREFERRED 类别，保证格式严格可验证。
    """
    module = _load_stage1_5_module()
    generators = module.GENERATORS
    categories = [c for c in STRICT_REPLAY_PREFERRED if c in generators]
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    attempts = 0
    max_attempts = count * 20
    while len(out) < count and attempts < max_attempts:
        attempts += 1
        category = rng.choice(categories)
        gen = generators[category]
        try:
            rows = gen("fresh", 1, rng)
        except Exception:  # noqa: BLE001
            continue
        if not rows:
            continue
        row = rows[0]
        try:
            user, assistant = get_user_and_assistant(row["messages"])
        except ValueError:
            continue
        h = prompt_hash(user)
        if h in seen:
            continue
        seen.add(h)
        row["source"] = source_tag
        out.append(row)
    log(f"generated {len(out)} fresh strict candidates (requested {count})")
    return out


def strict_replay_to_record(
    row: Mapping[str, Any],
    split: str,
    index: int,
    estimator: TokenEstimator,
) -> dict[str, Any]:
    user, assistant = get_user_and_assistant(row["messages"])
    total = estimator.count(user) + estimator.count(assistant)
    return {
        "id": f"stage2-{split}-stage1_5_strict_replay-{index:05d}",
        "category": "stage1_5_strict_replay",
        "source": "stage1_5_replay",
        "messages": copy.deepcopy(row["messages"]),
        "metadata": {
            "thinking": False,
            "final_answer": assistant,
            "reasoning_token_estimate": 0,
            "total_token_estimate": total,
            "source_original_id": str(row.get("id") or ""),
            "split": split,
            "source_category": row.get("category", ""),
        },
    }


# ---------------------------------------------------------------------------
# No Robots open replay（no-think，开放式回答）
# ---------------------------------------------------------------------------
def no_robots_open_candidate_ok(row: Mapping[str, Any]) -> bool:
    messages = row.get("messages")
    try:
        user, assistant = get_user_and_assistant(messages)
    except ValueError:
        return False
    if not user or not assistant:
        return False
    if row.get("category") not in OPEN_REPLAY_CATEGORIES:
        return False
    if len(user) >= 2500:
        return False
    count = word_count(assistant)
    if count < 50 or count > 250:
        return False
    lower = assistant.lstrip().lower()
    if lower.startswith(("i'm sorry", "i’m sorry", "i cannot")):
        return False
    if THINK_OPEN in assistant or THINK_CLOSE in assistant:
        return False
    return looks_clean_text(user) and looks_clean_text(assistant)


def load_no_robots_open_candidates(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    candidates = [row for row in rows if no_robots_open_candidate_ok(row)]
    log(f"no_robots open replay candidates from {path}: {len(candidates)}/{len(rows)}")
    return candidates


def open_replay_to_record(
    row: Mapping[str, Any],
    split: str,
    index: int,
    estimator: TokenEstimator,
) -> dict[str, Any]:
    user, assistant = get_user_and_assistant(row["messages"])
    total = estimator.count(user) + estimator.count(assistant)
    return {
        "id": f"stage2-{split}-no_robots_open_replay-{index:05d}",
        "category": "no_robots_open_replay",
        "source": row.get("source", "HuggingFaceH4/no_robots"),
        "messages": copy.deepcopy(row["messages"]),
        "metadata": {
            "thinking": False,
            "final_answer": "",
            "reasoning_token_estimate": 0,
            "total_token_estimate": total,
            "source_original_id": str(row.get("id") or row.get("prompt_id") or ""),
            "split": split,
            "source_category": row.get("category", ""),
        },
    }


# ---------------------------------------------------------------------------
# 候选池装配 + 全局 prompt-hash 去重
# ---------------------------------------------------------------------------
def _partition(items: Sequence[Mapping[str, Any]], ratios: Sequence[float]) -> list[list[Mapping[str, Any]]]:
    """按 ratios 比例将 items 切分为若干不相交子列表。"""
    total_ratio = sum(ratios)
    n = len(items)
    cuts: list[list[Mapping[str, Any]]] = []
    start = 0
    for i, r in enumerate(ratios):
        end = n if i == len(ratios) - 1 else int(round(n * r / total_ratio)) + start
        cuts.append(list(items[start:end]))
        start = end
    return cuts


def build_candidate_pools(
    splits: Sequence[str],
    args: argparse.Namespace,
    estimator: TokenEstimator,
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, dict[str, int]]]:
    """为每个 (split, category) 构建候选记录池（已转为最终 record 结构）。

    返回 (pools, available_counts)。``available_counts[split][category]`` 给出
    该 split 该类别当前可用的候选记录数（用于配额重分配，避免静默不足）。
    外部数据仅在显式提供 --cot-collection-path / --openthoughts-path / 对应
    hub-id 时才尝试加载，避免在 CPU 机器上意外触发大下载。
    """
    seed = args.seed
    programmatic_only = args.allow_programmatic_only
    cot_path = args.cot_collection_path
    open_path = args.openthoughts_path
    cot_hub = None if programmatic_only else args.cot_collection_hub_id
    open_hub = None if programmatic_only else args.openthoughts_hub_id

    # ---- 外部数据：加载 + 适配 + 按 train:val:test 比例切分不相交子集 ----
    cot_pool: dict[str, list[dict[str, Any]]] = {s: [] for s in splits}
    open_pool: dict[str, list[dict[str, Any]]] = {s: [] for s in splits}
    external_available = False

    # CoT-Collection（仅在提供 path 或 hub-id 时加载）
    if cot_path is not None or cot_hub is not None:
        if cot_hub is not None and cot_hub.lower() == "kaist-ai/cot-collection" and cot_path is None:
            # 专用 ijson 流式路径：load+去重+筛选一步完成，直接返回适配后候选
            # target 收集量 = 600(实际需要) * 冗余系数 4，保证 train:val:test=10:1:1 切分后各 split 充足
            cot_cands = _load_cot_collection_hub(
                target_count=2400,
                dataset_id=cot_hub,
                max_reasoning_tokens=300,
                max_answer_tokens=64,
                max_total_tokens=args.max_total_tokens,
                estimator=estimator,
            )
        else:
            # 本地 JSONL 路径 或 其他 CoT repo：走通用 load + adapter
            cot_records = load_external_records(
                "cot_collection", cot_path, cot_hub, "train", limit=40000,
            )
            cot_cands = []
            if cot_records:
                adapter = CotCollectionAdapter("cot_collection", cot_records, "cot_collection")
                cot_cands = adapter.adapt(
                    max_reasoning_tokens=300, max_answer_tokens=64,
                    max_total_tokens=args.max_total_tokens, estimator=estimator,
                )
        if cot_cands:
            rng_ext = random.Random(seed + 100)
            rng_ext.shuffle(cot_cands)
            # 比例 train:val:test = 500:50:50 = 10:1:1
            cot_parts = _partition(cot_cands, [10, 1, 1])
            for s, part in zip(splits, cot_parts):
                cot_pool[s] = part
            external_available = True
        else:
            log("cot_collection: 未取到任何记录，将跳过该外部源。")

    # OpenThoughts（仅在提供 path 或 hub-id 时加载）
    open_loaded = False
    if open_path is not None or open_hub is not None:
        open_records = load_external_records(
            "openthoughts", open_path, open_hub, "train", limit=40000,
        )
        if open_records:
            adapter = OpenThoughtsAdapter("openthoughts", open_records, "openthoughts")
            open_cands = adapter.adapt(
                max_reasoning_tokens=args.max_reasoning_tokens, max_answer_tokens=64,
                max_total_tokens=args.max_total_tokens, estimator=estimator,
            )
            rng_ext = random.Random(seed + 200)
            rng_ext.shuffle(open_cands)
            open_parts = _partition(open_cands, [10, 1, 1])
            for s, part in zip(splits, open_parts):
                open_pool[s] = part
            external_available = True
            open_loaded = True
        else:
            log("openthoughts: 未取到任何记录，将跳过该外部源。")

    if not external_available and not programmatic_only:
        raise ValueError(
            "外部数据源 CoT-Collection / OpenThoughts 均不可用，且未启用 --allow-programmatic-only。"
            "请提供 --cot-collection-path / --openthoughts-path（或对应 --*-hub-id），"
            "或加 --allow-programmatic-only 生成调试版本。"
        )
    if not external_available and programmatic_only:
        log("已启用 --allow-programmatic-only：外部 thinking 数据配额将置 0，仅生成 programmatic + replay。")

    pools: dict[str, dict[str, list[dict[str, Any]]]] = {s: {} for s in splits}
    available_counts: dict[str, dict[str, int]] = {s: {} for s in splits}

    for split in splits:
        # cot_collection_short
        cot_idx = 0
        cot_records_split = []
        for cand in cot_pool[split]:
            rec = external_candidate_to_record(
                cand, split, "cot_collection_short", "cot_collection", cot_idx, estimator
            )
            cot_records_split.append(rec)
            cot_idx += 1
        pools[split]["cot_collection_short"] = cot_records_split
        available_counts[split]["cot_collection_short"] = len(cot_records_split)

        # openthoughts_short
        open_idx = 0
        open_records_split = []
        for cand in open_pool[split]:
            rec = external_candidate_to_record(
                cand, split, "openthoughts_short", "openthoughts", open_idx, estimator
            )
            open_records_split.append(rec)
            open_idx += 1
        pools[split]["openthoughts_short"] = open_records_split
        available_counts[split]["openthoughts_short"] = len(open_records_split)

        # programmatic_thinking（每个 split 独立种子，避免 prompt 重复）
        subtype_counts = PROGRAMMATIC_SUBTYPES_TRAIN if split == "train" else PROGRAMMATIC_SUBTYPES_VAL
        prog_rng = random.Random(seed + {"train": 300, "validation": 301, "protocol_test": 302}[split])
        prog_cands = generate_programmatic_candidates(split, subtype_counts, prog_rng)
        prog_records = [
            programmatic_candidate_to_record(c, split, i, estimator)
            for i, c in enumerate(prog_cands)
        ]
        pools[split]["programmatic_thinking"] = prog_records
        available_counts[split]["programmatic_thinking"] = len(prog_records)

        # stage1_5_strict_replay：train <- stage1_5 train；
        # val/test <- stage1_5 validation + 重新生成的未见 strict 样本（两半不相交）
        if split == "train":
            strict_src = load_stage1_5_strict_candidates(args.stage1_5_train)
            strict_rng = random.Random(seed + 400)
            strict_rng.shuffle(strict_src)
            strict_pool = strict_src
        else:
            strict_src = load_stage1_5_strict_candidates(args.stage1_5_validation)
            # 补足：重新生成未见 strict 样本，避免与 train 的模板 prompt 冲突后配额不足
            fresh_rng = random.Random(seed + 402 if split == "validation" else seed + 403)
            fresh = generate_fresh_strict_candidates(220, fresh_rng)
            combined = strict_src + fresh
            combined_rng = random.Random(seed + 401)
            combined_rng.shuffle(combined)
            half = len(combined) // 2
            strict_pool = combined[:half] if split == "validation" else combined[half:]
        strict_records = [
            strict_replay_to_record(row, split, i, estimator) for i, row in enumerate(strict_pool)
        ]
        pools[split]["stage1_5_strict_replay"] = strict_records
        available_counts[split]["stage1_5_strict_replay"] = len(strict_records)

        # no_robots_open_replay：train <- stage1 train；val/test <- stage1 validation 两半
        if split == "train":
            open_src = load_no_robots_open_candidates(args.stage1_train)
            open_rng = random.Random(seed + 500)
            open_rng.shuffle(open_src)
            open_pool_src = open_src
        else:
            open_src = load_no_robots_open_candidates(args.stage1_validation)
            open_rng = random.Random(seed + 501)
            open_rng.shuffle(open_src)
            half = len(open_src) // 2
            open_pool_src = open_src[:half] if split == "validation" else open_src[half:]
        open_replay_records = [
            open_replay_to_record(row, split, i, estimator) for i, row in enumerate(open_pool_src)
        ]
        pools[split]["no_robots_open_replay"] = open_replay_records
        available_counts[split]["no_robots_open_replay"] = len(open_replay_records)

    return pools, available_counts


PROGRAMMATIC_SUBTYPE_QUOTAS: dict[str, Mapping[str, int]] = {
    "train": PROGRAMMATIC_SUBTYPES_TRAIN,
    "validation": PROGRAMMATIC_SUBTYPES_VAL,
    "protocol_test": PROGRAMMATIC_SUBTYPES_VAL,
}


def _regenerate_programmatic_pool(
    pools: dict[str, dict[str, list[dict[str, Any]]]],
    split: str,
    subtype_counts: Mapping[str, int],
    seed: int,
    estimator: TokenEstimator,
) -> None:
    """当 programmatic 配额被上调（外部缺口重分配）时，按新子类配额重新生成候选池。

    用 split 专属种子重新生成，覆盖原 programmatic_thinking 池。
    """
    prog_rng = random.Random(seed + {"train": 300, "validation": 301, "protocol_test": 302}[split])
    prog_cands = generate_programmatic_candidates(split, subtype_counts, prog_rng)
    prog_records = [
        programmatic_candidate_to_record(c, split, i, estimator)
        for i, c in enumerate(prog_cands)
    ]
    pools[split]["programmatic_thinking"] = prog_records
    log(f"{split}: programmatic pool regenerated with {len(prog_records)} candidates "
        f"(subtypes={dict(subtype_counts)})")


def select_with_global_dedup(
    pools: Mapping[str, Mapping[str, list[dict[str, Any]]]],
    quotas_by_split: Mapping[str, Mapping[str, int]],
    splits: Sequence[str],
    subtype_quotas_by_split: Mapping[str, Mapping[str, int]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """按 train -> validation -> protocol_test 顺序选取，全局 prompt-hash 去重。

    programmatic_thinking 按子类配额选取，确保每个子类都被代表（避免小配额 split
    上某子类被随机漏掉）。子类配额由 ``subtype_quotas_by_split`` 提供，缺省回退到
    ``PROGRAMMATIC_SUBTYPE_QUOTAS``。
    """
    seen_hashes: dict[str, str] = {}
    selected: dict[str, list[dict[str, Any]]] = {}

    def _pick_from(pool: Sequence[dict[str, Any]], quota: int, seen_hashes: dict[str, str], split: str) -> list[dict[str, Any]]:
        picked: list[dict[str, Any]] = []
        for rec in pool:
            user, _ = get_user_and_assistant(rec["messages"])
            h = prompt_hash(user)
            if h in seen_hashes:
                continue
            seen_hashes[h] = split
            picked.append(rec)
            if len(picked) >= quota:
                break
        return picked

    for split in splits:
        quotas = quotas_by_split[split]
        split_rows: list[dict[str, Any]] = []
        shortages: list[str] = []
        for category in CATEGORY_ORDER:
            quota = quotas.get(category, 0)
            if quota == 0:
                continue
            pool = pools[split].get(category, [])
            if category == "programmatic_thinking":
                # 按子类配额选取，确保各子类代表
                subtype_quotas = (
                    subtype_quotas_by_split[split]
                    if subtype_quotas_by_split is not None
                    else PROGRAMMATIC_SUBTYPE_QUOTAS[split]
                )
                subtype_pools: dict[str, list[dict[str, Any]]] = {s: [] for s in subtype_quotas}
                for rec in pool:
                    st = rec.get("metadata", {}).get("subtype", "")
                    if st in subtype_pools:
                        subtype_pools[st].append(rec)
                total_picked = 0
                for subtype, sub_quota in subtype_quotas.items():
                    sub_pool = subtype_pools.get(subtype, [])
                    picked = _pick_from(sub_pool, sub_quota, seen_hashes, split)
                    split_rows.extend(picked)
                    total_picked += len(picked)
                    if len(picked) < sub_quota:
                        shortages.append(f"{split}/programmatic_thinking/{subtype}: {len(picked)}/{sub_quota}")
                if total_picked < quota:
                    shortages.append(f"{split}/programmatic_thinking: {total_picked}/{quota}")
            else:
                picked = _pick_from(pool, quota, seen_hashes, split)
                split_rows.extend(picked)
                if len(picked) < quota:
                    shortages.append(f"{split}/{category}: {len(picked)}/{quota}")
        if shortages:
            raise ValueError(
                "候选不足，无法满足配额（不要静默生成数量不足的数据）:\n  " + "\n  ".join(shortages)
            )
        selected[split] = split_rows
    return selected


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------
def _record_id_uniqueness(all_rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    ids: set[str] = set()
    for split, rows in all_rows.items():
        for index, row in enumerate(rows):
            rid = row.get("id")
            if not isinstance(rid, str) or not rid:
                raise ValueError(f"{split}[{index}] missing nonempty id")
            if rid in ids:
                raise ValueError(f"duplicate id across splits: {rid}")
            ids.add(rid)


def validate_base_record(row: Mapping[str, Any], split: str, index: int) -> None:
    loc = f"{split}[{index}]"
    for field in ("id", "category", "source", "messages", "metadata"):
        if field not in row:
            raise ValueError(f"{loc} missing field: {field}")
    if not isinstance(row["id"], str) or not row["id"]:
        raise ValueError(f"{loc} id must be nonempty string")
    if not isinstance(row["category"], str) or row["category"] not in TRAIN_QUOTAS:
        raise ValueError(f"{loc} unsupported category: {row.get('category')!r}")
    if not isinstance(row["source"], str) or not row["source"]:
        raise ValueError(f"{loc} source must be nonempty string")
    user, assistant = get_user_and_assistant(row["messages"])
    if not user:
        raise ValueError(f"{loc} user content empty")
    if not assistant:
        raise ValueError(f"{loc} assistant content empty")
    if "<|im_end|>" in assistant:
        raise ValueError(f"{loc} assistant must not contain <|im_end|>")
    meta = row.get("metadata")
    if not isinstance(meta, Mapping):
        raise ValueError(f"{loc} metadata must be an object")


def validate_thinking_record(row: Mapping[str, Any], split: str, index: int, max_reasoning_tokens: int, max_total_tokens: int, estimator: TokenEstimator) -> None:
    loc = f"{split}[{index}]"
    _, assistant = get_user_and_assistant(row["messages"])
    parsed = parse_thinking(assistant)
    if parsed is None:
        raise ValueError(f"{loc} thinking sample missing valid <think>...</think> structure")
    reasoning, final = parsed
    if not final:
        raise ValueError(f"{loc} final answer after </think> is empty")
    meta = row["metadata"]
    if meta.get("thinking") is not True:
        raise ValueError(f"{loc} thinking category must have metadata.thinking=true")
    if meta.get("final_answer") != final:
        raise ValueError(f"{loc} metadata.final_answer mismatch with parsed final")
    r_tokens = estimator.count(reasoning)
    if r_tokens > max_reasoning_tokens + 50:
        raise ValueError(f"{loc} reasoning exceeds max_reasoning_tokens: {r_tokens}")
    total = estimator.count(row["messages"][0]["content"]) + estimator.count(assistant)
    if total > max_total_tokens + 128:
        raise ValueError(f"{loc} total exceeds max_total_tokens: {total}")
    # final answer 后不应继续长篇解释：final 不应包含换行后的长段
    if "\n" in final and word_count(final) > 40:
        raise ValueError(f"{loc} final answer looks like a long explanation")
    subtype = meta.get("subtype", "")
    if subtype == "binary_operation_thinking" or subtype == "wonderland_like_binary_thinking":
        if not re.fullmatch(r"[01]{8}", final):
            raise ValueError(f"{loc} binary final answer must match [01]{{8}}: {final!r}")
    if subtype == "json_final_thinking":
        try:
            json.loads(final)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{loc} json final answer must be valid JSON: {exc}") from exc
    if subtype == "yes_no_reasoning_thinking":
        if final not in {"yes", "no"}:
            raise ValueError(f"{loc} yes_no final must be yes/no: {final!r}")
    if subtype == "classification_thinking":
        if final not in CLASSIFICATION_LABELS:
            raise ValueError(f"{loc} classification final must be a label: {final!r}")


def validate_nothink_record(row: Mapping[str, Any], split: str, index: int) -> None:
    loc = f"{split}[{index}]"
    _, assistant = get_user_and_assistant(row["messages"])
    if THINK_OPEN in assistant or THINK_CLOSE in assistant:
        raise ValueError(f"{loc} no-think replay must not contain thinking tags")
    meta = row["metadata"]
    if meta.get("thinking") is not False:
        raise ValueError(f"{loc} no-think category must have metadata.thinking=false")
    category = row["category"]
    if category == "stage1_5_strict_replay":
        src_cat = meta.get("source_category", "")
        stripped = assistant.strip()
        if src_cat in {"binary_only", "wonderland_like_binary"} and not re.fullmatch(r"[01]{8}", stripped):
            raise ValueError(f"{loc} strict replay binary mismatch: {stripped!r}")
        if src_cat == "json_only":
            try:
                json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{loc} strict replay json invalid: {exc}") from exc
        if src_cat == "yes_no" and stripped not in {"yes", "no"}:
            raise ValueError(f"{loc} strict replay yes_no mismatch: {stripped!r}")
    elif category == "no_robots_open_replay":
        wc = word_count(assistant)
        if wc < 50 or wc > 250:
            raise ValueError(f"{loc} no_robots open replay word count out of [50,250]: {wc}")


def validate_split(rows: Sequence[Mapping[str, Any]], split: str, quotas: Mapping[str, int], max_reasoning_tokens: int, max_total_tokens: int, estimator: TokenEstimator) -> None:
    counts = Counter(row["category"] for row in rows)
    for category, quota in quotas.items():
        if counts.get(category, 0) != quota:
            raise ValueError(
                f"{split} category {category} count {counts.get(category, 0)} != quota {quota}"
            )
    if len(rows) != sum(quotas.values()):
        raise ValueError(f"{split} total count {len(rows)} != expected {sum(quotas.values())}")
    for index, row in enumerate(rows):
        validate_base_record(row, split, index)
        if row["category"] in THINKING_CATEGORIES:
            validate_thinking_record(row, split, index, max_reasoning_tokens, max_total_tokens, estimator)
        else:
            validate_nothink_record(row, split, index)


def validate_cross_split_dedup(all_rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    seen: dict[str, str] = {}
    for split, rows in all_rows.items():
        for index, row in enumerate(rows):
            user, _ = get_user_and_assistant(row["messages"])
            h = prompt_hash(user)
            if h in seen and seen[h] != split:
                raise ValueError(
                    f"prompt hash collision across splits: {h[:12]} in {seen[h]} and {split} "
                    f"(id={row['id']})"
                )
            seen[h] = split


# ---------------------------------------------------------------------------
# 统计与样本展示
# ---------------------------------------------------------------------------
def _length_stats(values: Sequence[int]) -> dict[str, float]:
    if not values:
        return {"min": 0, "max": 0, "mean": 0}
    return {
        "min": min(values),
        "max": max(values),
        "mean": round(statistics.fmean(values), 2),
    }


def compute_stats(rows: Sequence[Mapping[str, Any]], estimator: TokenEstimator) -> dict[str, Any]:
    cat_counts = Counter(row["category"] for row in rows)
    thinking = sum(1 for row in rows if row["metadata"].get("thinking") is True)
    nothink = len(rows) - thinking
    strict = cat_counts.get("stage1_5_strict_replay", 0)
    open_rep = cat_counts.get("no_robots_open_replay", 0)
    external_cot = cat_counts.get("cot_collection_short", 0)
    programmatic = cat_counts.get("programmatic_thinking", 0)
    totals = [row["metadata"].get("total_token_estimate", 0) for row in rows]
    reasoning = [
        row["metadata"].get("reasoning_token_estimate", 0)
        for row in rows
        if row["metadata"].get("thinking") is True
    ]
    return {
        "count": len(rows),
        "categories": dict(sorted(cat_counts.items())),
        "thinking": thinking,
        "nothinking": nothink,
        "strict_replay": strict,
        "open_replay": open_rep,
        "external_cot": external_cot,
        "programmatic": programmatic,
        "total_tokens": _length_stats(totals),
        "reasoning_tokens": _length_stats(reasoning),
    }


def print_report(all_rows: Mapping[str, Sequence[Mapping[str, Any]]], estimator: TokenEstimator, args: argparse.Namespace) -> None:
    log("=== Stage 2 数据报告 ===")
    for split, rows in all_rows.items():
        stats = compute_stats(rows, estimator)
        log(f"-- {split} --")
        log(f"  count: {stats['count']}")
        log(f"  categories: {stats['categories']}")
        log(f"  thinking={stats['thinking']} ({round(100*stats['thinking']/stats['count'],1)}%) "
            f"no-thinking={stats['nothinking']} ({round(100*stats['nothinking']/stats['count'],1)}%)")
        log(f"  strict_replay={stats['strict_replay']} open_replay={stats['open_replay']} "
            f"external_cot={stats['external_cot']} programmatic={stats['programmatic']}")
        log(f"  total_token_estimate: {stats['total_tokens']}")
        log(f"  reasoning_token_estimate (thinking only): {stats['reasoning_tokens']}")
    # 比例汇总（train）
    train_rows = all_rows["train"]
    train_stats = compute_stats(train_rows, estimator)
    log(
        f"train 比例: thinking={round(100*train_stats['thinking']/len(train_rows),1)}% "
        f"no-thinking={round(100*train_stats['nothinking']/len(train_rows),1)}% "
        f"(目标 thinking≈60%, no-thinking≈40%)"
    )


def print_samples(all_rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    log("=== 每大类 2 条样本展示 ===")
    for category in CATEGORY_ORDER:
        log(f"-- {category} --")
        shown = 0
        for split in all_rows:
            for row in all_rows[split]:
                if row["category"] != category:
                    continue
                user, assistant = get_user_and_assistant(row["messages"])
                log(f"  id={row['id']}")
                log(f"  category={row['category']} source={row['source']} split={split}")
                log(f"  thinking={row['metadata'].get('thinking')}")
                log(f"  final_answer={row['metadata'].get('final_answer','')[:200]!r}")
                log(f"  user[:300]={user[:300]!r}")
                log(f"  assistant[:500]={assistant[:500]!r}")
                shown += 1
                break
            if shown >= 2:
                break
            # 取第二个样本：允许跨 split
        if shown < 2:
            # 尝试在同一 split 再取一条
            for split in all_rows:
                count = 0
                for row in all_rows[split]:
                    if row["category"] != category:
                        continue
                    count += 1
                    if count <= shown:
                        continue
                    user, assistant = get_user_and_assistant(row["messages"])
                    log(f"  id={row['id']} (2nd)")
                    log(f"  category={row['category']} source={row['source']} split={split}")
                    log(f"  thinking={row['metadata'].get('thinking')}")
                    log(f"  final_answer={row['metadata'].get('final_answer','')[:200]!r}")
                    log(f"  user[:300]={user[:300]!r}")
                    log(f"  assistant[:500]={assistant[:500]!r}")
                    shown += 1
                    break
                if shown >= 2:
                    break


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Stage 2 thinking-format warmup JSONL data."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("data/instruction/stage2_thinking"))
    parser.add_argument("--cot-collection-path", type=Path, default=None)
    parser.add_argument("--openthoughts-path", type=Path, default=None)
    parser.add_argument(
        "--cot-collection-hub-id",
        type=str,
        default=None,
        help="可选：从 HuggingFace Hub 下载 CoT-Collection 的 dataset id（显式 opt-in，避免意外大下载）",
    )
    parser.add_argument(
        "--openthoughts-hub-id",
        type=str,
        default=None,
        help="可选：从 HuggingFace Hub 下载 OpenThoughts 的 dataset id（显式 opt-in）",
    )
    parser.add_argument("--stage1-5-train", type=Path, default=DEFAULT_STAGE1_5_TRAIN)
    parser.add_argument("--stage1-5-validation", type=Path, default=DEFAULT_STAGE1_5_VALIDATION)
    parser.add_argument("--stage1-train", type=Path, default=DEFAULT_STAGE1_TRAIN)
    parser.add_argument("--stage1-validation", type=Path, default=DEFAULT_STAGE1_VALIDATION)
    parser.add_argument("--max-total-tokens", type=int, default=1536)
    parser.add_argument("--max-reasoning-tokens", type=int, default=400)
    parser.add_argument("--dry-run", action="store_true", help="只打印统计，不写入文件")
    parser.add_argument(
        "--allow-programmatic-only",
        action="store_true",
        help="外部数据不可用时，只生成 programmatic + replay 的调试版本",
    )
    return parser.parse_args()


def effective_quotas(
    args: argparse.Namespace,
    external_available: bool,
    available_counts: Mapping[str, Mapping[str, int]] | None = None,
) -> tuple[dict[str, Mapping[str, int]], dict[str, Mapping[str, int]]]:
    """根据外部数据可用性返回每个 split 的有效配额与 programmatic 子类配额。

    若某外部 thinking 源（cot_collection_short / openthoughts_short）候选不足原配额，
    将缺口重分配到 programmatic_thinking（及其子类），保持总量不变且不静默不足。

    返回 (quotas_by_split, subtype_quotas_by_split)。
    """
    base = {
        "train": dict(TRAIN_QUOTAS),
        "validation": dict(VALIDATION_QUOTAS),
        "protocol_test": dict(PROTOCOL_TEST_QUOTAS),
    }
    if not external_available:
        # programmatic-only 调试版本：外部 thinking 配额（cot / openthoughts）置 0
        for split in base:
            for k in list(base[split]):
                if k in THINKING_CATEGORIES and k != "programmatic_thinking":
                    base[split][k] = 0
        subtype_quotas = {
            split: dict(PROGRAMMATIC_SUBTYPE_QUOTAS[split]) for split in base
        }
        return base, subtype_quotas

    # 外部可用：按 available_counts 把不足的外部配额重分配到 programmatic_thinking
    subtype_quotas: dict[str, dict[str, int]] = {}
    for split, quotas in base.items():
        avail = available_counts.get(split, {}) if available_counts else {}
        shortfall = 0
        for ext_cat in ("cot_collection_short", "openthoughts_short"):
            want = quotas[ext_cat]
            have = min(want, avail.get(ext_cat, 0))
            if have < want:
                shortfall += want - have
                quotas[ext_cat] = have
                log(f"{split}/{ext_cat}: 候选 {have} < 配额 {want}，缺口 {want-have} 重分配到 programmatic_thinking")
            else:
                quotas[ext_cat] = want
        if shortfall > 0:
            quotas["programmatic_thinking"] = quotas["programmatic_thinking"] + shortfall
            log(f"{split}/programmatic_thinking: 配额上调 +{shortfall} -> {quotas['programmatic_thinking']}")
        # 同步子类配额：按 programmatic 总配额等比放大原有子类比例
        base_subtype = dict(PROGRAMMATIC_SUBTYPE_QUOTAS[split])
        base_subtotal = sum(base_subtype.values())
        if quotas["programmatic_thinking"] != base_subtotal and base_subtotal > 0:
            scale = quotas["programmatic_thinking"] / base_subtotal
            scaled = {k: max(1, int(round(v * scale))) for k, v in base_subtype.items()}
            # 修正取整误差，使子类和精确等于总配额
            diff = quotas["programmatic_thinking"] - sum(scaled.values())
            if diff != 0:
                # 把误差加到最大的子类（binary_operation / wonderland_like_binary）
                top = max(scaled, key=lambda k: scaled[k])
                scaled[top] += diff
            subtype_quotas[split] = scaled
        else:
            subtype_quotas[split] = base_subtype
    return base, subtype_quotas


def main() -> None:
    args = parse_args()
    log(f"seed={args.seed} output_dir={args.output_dir} dry_run={args.dry_run} "
        f"programmatic_only={args.allow_programmatic_only}")
    for label, p in [
        ("stage1_5_train", args.stage1_5_train),
        ("stage1_5_validation", args.stage1_5_validation),
        ("stage1_train", args.stage1_train),
        ("stage1_validation", args.stage1_validation),
    ]:
        if not p.exists():
            raise ValueError(f"{label} not found: {p}")

    estimator = TokenEstimator(cache_dir=Path(".hf-cache/hub"))
    splits = ["train", "validation", "protocol_test"]

    # 构建候选池（build_candidate_pools 在外部不可用且未启用 programmatic-only 时会抛错）
    pools, available_counts = build_candidate_pools(splits, args, estimator)
    external_available = any(available_counts[s].get("cot_collection_short", 0) > 0 or available_counts[s].get("openthoughts_short", 0) > 0 for s in splits)

    if not external_available and not args.allow_programmatic_only:
        raise ValueError(
            "外部数据不可用且未启用 --allow-programmatic-only。请提供外部数据路径或启用调试模式。"
        )

    quotas_by_split, subtype_quotas_by_split = effective_quotas(args, external_available, available_counts)
    log(f"effective quotas: train={dict(quotas_by_split['train'])}")

    # 若 programmatic 配额被上调（外部缺口重分配），需按新子类配额补生成候选
    for split in splits:
        new_prog_quota = quotas_by_split[split]["programmatic_thinking"]
        base_prog_quota = PROGRAMMATIC_SUBTYPE_QUOTAS[split]
        if sum(subtype_quotas_by_split[split].values()) > sum(base_prog_quota.values()):
            _regenerate_programmatic_pool(pools, split, subtype_quotas_by_split[split], args.seed, estimator)

    selected = select_with_global_dedup(pools, quotas_by_split, splits, subtype_quotas_by_split)

    # 校验
    _record_id_uniqueness(selected)
    validate_cross_split_dedup(selected)
    for split in splits:
        validate_split(selected[split], split, quotas_by_split[split], args.max_reasoning_tokens, args.max_total_tokens, estimator)
    log("所有数据质量检查通过。")

    # 报告 + 样本展示
    print_report(selected, estimator, args)
    print_samples(selected)

    # 确认声明
    log("确认：未使用官方 Wonderland validation/test 数据。")
    log("确认：未启动训练，未修改 train_sft / 模型加载 / LoRA / chat template。")
    log("确认：assistant content 未写入 <|im_end|>。")

    if args.dry_run:
        log("--dry-run 模式：不写入文件。")
        return

    for split in splits:
        out_path = args.output_dir / f"{split}.jsonl"
        write_jsonl_atomic(out_path, selected[split])
        log(f"wrote {out_path} ({len(selected[split])} rows)")
    log(f"输出目录：{args.output_dir}")


if __name__ == "__main__":
    main()
