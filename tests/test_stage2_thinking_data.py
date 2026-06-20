import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "generate_stage2_thinking_data.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stage2_thinking_data", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _open_replay_row(index: int, *, category: str = "Open QA", words: int = 70) -> dict:
    body = " ".join(f"word{index}_{i}" for i in range(words))
    return {
        "id": f"stage1-{index}",
        "category": category,
        "messages": [
            {"role": "user", "content": f"Question {index}?"},
            {"role": "assistant", "content": body},
        ],
        "source": "HuggingFaceH4/no_robots",
        "source_split": "train",
        "source_row_index": index,
    }


def _strict_row(index: int, *, category: str = "binary_only", answer: str = "10101101") -> dict:
    return {
        "id": f"stage1_5-train-{category}-{index:05d}",
        "category": category,
        "messages": [
            {"role": "user", "content": f"Return only this 8-bit binary string and nothing else: {answer}."},
            {"role": "assistant", "content": answer},
        ],
        "validator": {"type": "regex", "pattern": "^[01]{8}$"},
    }


def _make_args(module, tmp_path, *, programmatic_only=True):
    ns = type("NS", (), {})()
    ns.seed = 42
    ns.output_dir = tmp_path / "stage2_thinking"
    ns.cot_collection_path = None
    ns.openthoughts_path = None
    ns.cot_collection_hub_id = None
    ns.openthoughts_hub_id = None
    ns.allow_programmatic_only = programmatic_only
    ns.max_total_tokens = 1536
    ns.max_reasoning_tokens = 400
    ns.dry_run = False
    return ns


def _prepare_replay_inputs(tmp_path: Path, module) -> dict:
    """写最小的 stage1 / stage1_5 输入文件，供 replay 使用。"""
    stage1_5_train = tmp_path / "stage1_5" / "train.jsonl"
    stage1_5_val = tmp_path / "stage1_5" / "validation.jsonl"
    stage1_train = tmp_path / "stage1" / "train.jsonl"
    stage1_val = tmp_path / "stage1" / "validation.jsonl"

    # stage1_5 train: 多种 strict 类别，足够 train replay 500 条
    s15_train_rows = []
    for i in range(200):
        s15_train_rows.append(_strict_row(i, category="binary_only", answer=format(i % 256, "08b")))
    for i in range(200):
        s15_train_rows.append(_strict_row(200 + i, category="exact_output", answer="OK"))
    for i in range(200):
        s15_train_rows.append(_strict_row(400 + i, category="yes_no", answer="yes" if i % 2 else "no"))
    for i in range(200):
        s15_train_rows.append(_strict_row(600 + i, category="wonderland_like_binary", answer=format(i % 256, "08b")))
    write_jsonl(stage1_5_train, s15_train_rows)

    # stage1_5 validation: 供 val/test replay
    s15_val_rows = []
    for i in range(80):
        s15_val_rows.append(_strict_row(i, category="binary_only", answer=format((i + 7) % 256, "08b")))
    for i in range(80):
        s15_val_rows.append(_strict_row(80 + i, category="exact_output", answer="PASS"))
    write_jsonl(stage1_5_val, s15_val_rows)

    # stage1 train/val: open replay
    write_jsonl(stage1_train, [_open_replay_row(i, category="Open QA") for i in range(800)])
    write_jsonl(stage1_val, [_open_replay_row(i, category="Generation", words=80) for i in range(120)])

    return {
        "stage1_5_train": stage1_5_train,
        "stage1_5_validation": stage1_5_val,
        "stage1_train": stage1_train,
        "stage1_validation": stage1_val,
    }


def test_thinking_tag_bytes_are_qwen3_standard():
    """thinking 标签必须是 Qwen3 标准 MENT / MENT（纯 ASCII）。"""
    module = load_module()
    assert module.THINK_OPEN.encode("utf-8") == b"\x3c\x74\x68\x69\x6e\x6b\x3e"
    assert module.THINK_CLOSE.encode("utf-8") == b"\x3c\x2f\x74\x68\x69\x6e\x6b\x3e"


def test_build_thinking_assistant_structure():
    module = load_module()
    content = module.build_thinking_assistant("step 1\nstep 2", "42")
    assert content.startswith(module.THINK_OPEN + "\n")
    assert module.THINK_CLOSE in content
    parsed = module.parse_thinking(content)
    assert parsed is not None
    reasoning, final = parsed
    assert reasoning == "step 1\nstep 2"
    assert final == "42"


def test_parse_thinking_rejects_malformed():
    module = load_module()
    assert module.parse_thinking("no tags here") is None
    # 闭标签在开标签之前
    bad = module.THINK_CLOSE + module.THINK_OPEN
    assert module.parse_thinking(bad) is None
    # 空推理
    empty_reasoning = module.THINK_OPEN + "\n" + module.THINK_CLOSE + "\n\n42"
    assert module.parse_thinking(empty_reasoning) is None


def test_programmatic_generators_produce_valid_thinking_format():
    module = load_module()
    rng = module.random.Random(0)
    for subtype, gen in module.PROGRAMMATIC_GENERATORS.items():
        sample = gen(rng)
        assert "user" in sample and "reasoning" in sample and "final_answer" in sample
        assert sample["reasoning"].strip() == sample["reasoning"]
        assert sample["final_answer"].strip() == sample["final_answer"]
        # binary 子类 final 必须是 8-bit
        if subtype in {"binary_operation_thinking", "wonderland_like_binary_thinking"}:
            import re

            assert re.fullmatch(r"[01]{8}", sample["final_answer"])
        if subtype == "json_final_thinking":
            json.loads(sample["final_answer"])
        if subtype == "yes_no_reasoning_thinking":
            assert sample["final_answer"] in {"yes", "no"}


def test_external_adapter_field_detection_and_adaptation():
    """适配层应能从异构字段名抽取 question/reasoning/answer，并通过 CoT 筛选。"""
    module = load_module()
    long_reason = (
        "To solve this, first recall the relevant fact, then apply it step by step, "
        "checking each intermediate result before stating the final answer clearly. "
        "This reasoning is long enough to pass the minimum token threshold."
    )
    records = [
        {"instruction": "What is twenty plus twenty-two?", "rationale": long_reason, "answer": "42"},
        {"question": "What is the capital city of France?", "cot": long_reason, "target": "Paris"},
        {"problem": "What is five multiplied by six?", "reasoning": long_reason, "output": "30"},
        {"prompt": "bad", "reasoning": "", "answer": ""},  # 应被丢弃（空字段）
    ]
    estimator = module.TokenEstimator.__new__(module.TokenEstimator)
    estimator._tokenizer = None
    estimator._tried = True
    estimator._cache_dir = None
    adapter = module.CotCollectionAdapter("cot_collection", records, "cot_collection")
    cands = adapter.adapt(
        max_reasoning_tokens=300, max_answer_tokens=64, max_total_tokens=1536, estimator=estimator,
    )
    # 三条有效记录被适配并通过筛选，一条空字段被丢弃
    assert len(cands) == 3
    questions = {c["question"] for c in cands}
    assert "What is twenty plus twenty-two?" in questions
    assert "What is the capital city of France?" in questions
    # 字段探测记录了实际使用的字段名
    used = {c["fields_used"]["question"] for c in cands}
    assert {"instruction", "question", "problem"} <= used
    used_reason = {c["fields_used"]["reasoning"] for c in cands}
    assert {"rationale", "cot", "reasoning"} <= used_reason


def test_external_adapter_rejects_open_ended_and_long_answers():
    """CoT 筛选应拒绝开放作文题和过长 final answer。"""
    module = load_module()
    long_reason = "x " * 120  # 足够长
    records = [
        # 开放作文题
        {"question": "Write an essay about the industrial revolution.", "rationale": long_reason, "answer": "essay text"},
        # final answer 过长
        {"question": "What is the meaning of life?", "rationale": long_reason,
         "answer": "a " * 200},
    ]
    estimator = module.TokenEstimator.__new__(module.TokenEstimator)
    estimator._tokenizer = None
    estimator._tried = True
    estimator._cache_dir = None
    adapter = module.CotCollectionAdapter("cot_collection", records, "cot_collection")
    cands = adapter.adapt(
        max_reasoning_tokens=300, max_answer_tokens=64, max_total_tokens=1536, estimator=estimator,
    )
    assert cands == []


def test_programmatic_only_generation_counts_and_validation(tmp_path):
    module = load_module()
    args = _make_args(module, tmp_path, programmatic_only=True)
    paths = _prepare_replay_inputs(tmp_path, module)
    args.stage1_5_train = paths["stage1_5_train"]
    args.stage1_5_validation = paths["stage1_5_validation"]
    args.stage1_train = paths["stage1_train"]
    args.stage1_validation = paths["stage1_validation"]

    # 缩小配额以适配测试输入规模
    small_train = {**module.TRAIN_QUOTAS,
                   "cot_collection_short": 0, "openthoughts_short": 0,
                   "programmatic_thinking": 40, "stage1_5_strict_replay": 50,
                   "no_robots_open_replay": 30}
    small_val = {**module.VALIDATION_QUOTAS,
                 "cot_collection_short": 0, "openthoughts_short": 0,
                 "programmatic_thinking": 40, "stage1_5_strict_replay": 50,
                 "no_robots_open_replay": 30}
    # 缩小 programmatic 子类配额（总和需等于 40），适配 per-subtype 选取逻辑
    small_subtypes = {
        "simple_math_thinking": 6,
        "comparison_thinking": 6,
        "yes_no_reasoning_thinking": 6,
        "binary_operation_thinking": 6,
        "json_final_thinking": 6,
        "classification_thinking": 5,
        "wonderland_like_binary_thinking": 5,
    }
    # monkeypatch 配额常量
    original_train = module.TRAIN_QUOTAS
    original_val = module.VALIDATION_QUOTAS
    original_pt = module.PROTOCOL_TEST_QUOTAS
    original_subtypes = module.PROGRAMMATIC_SUBTYPE_QUOTAS
    module.TRAIN_QUOTAS = small_train
    module.VALIDATION_QUOTAS = small_val
    module.PROTOCOL_TEST_QUOTAS = small_val
    module.PROGRAMMATIC_SUBTYPE_QUOTAS = {
        "train": small_subtypes, "validation": small_subtypes, "protocol_test": small_subtypes,
    }
    try:
        estimator = module.TokenEstimator.__new__(module.TokenEstimator)
        estimator._tokenizer = None
        estimator._tried = True
        estimator._cache_dir = None
        splits = ["train", "validation", "protocol_test"]
        pools, available_counts = module.build_candidate_pools(splits, args, estimator)
        external_available = any(
            available_counts[s].get("cot_collection_short", 0) > 0
            or available_counts[s].get("openthoughts_short", 0) > 0
            for s in splits
        )
        assert external_available is False
        quotas_by_split, subtype_quotas_by_split = module.effective_quotas(
            args, external_available, available_counts
        )
        selected = module.select_with_global_dedup(
            pools, quotas_by_split, splits, subtype_quotas_by_split
        )
    finally:
        module.TRAIN_QUOTAS = original_train
        module.VALIDATION_QUOTAS = original_val
        module.PROTOCOL_TEST_QUOTAS = original_pt
        module.PROGRAMMATIC_SUBTYPE_QUOTAS = original_subtypes

    # 配额满足
    for split in splits:
        counts = Counter(r["category"] for r in selected[split])
        for cat, q in quotas_by_split[split].items():
            assert counts[cat] == q, f"{split}/{cat}: {counts[cat]} != {q}"

    # id 全局唯一
    all_ids = [r["id"] for split in splits for r in selected[split]]
    assert len(set(all_ids)) == len(all_ids)
    # 跨 split prompt-hash 去重
    module.validate_cross_split_dedup(selected)
    # 质量校验通过
    for split in splits:
        module.validate_split(selected[split], split, quotas_by_split[split],
                              args.max_reasoning_tokens, args.max_total_tokens, estimator)


def test_insufficient_candidates_raises_without_silent_write(tmp_path):
    """候选不足时必须报错，不得静默写文件。"""
    module = load_module()
    pools = {
        "train": {"programmatic_thinking": [], "stage1_5_strict_replay": [],
                  "no_robots_open_replay": [], "cot_collection_short": [], "openthoughts_short": []},
        "validation": {c: [] for c in module.TRAIN_QUOTAS},
        "protocol_test": {c: [] for c in module.TRAIN_QUOTAS},
    }
    quotas = {"train": {"programmatic_thinking": 10}, "validation": {}, "protocol_test": {}}
    with pytest.raises(ValueError, match="候选不足"):
        module.select_with_global_dedup(pools, quotas, ["train", "validation", "protocol_test"])


def test_no_think_replay_must_not_contain_thinking_tags():
    module = load_module()
    row = {
        "id": "x",
        "category": "stage1_5_strict_replay",
        "source": "stage1_5_replay",
        "messages": [
            {"role": "user", "content": "Return only 10101101."},
            {"role": "assistant", "content": module.THINK_OPEN + " x " + module.THINK_CLOSE + " 10101101"},
        ],
        "metadata": {"thinking": False, "final_answer": "10101101",
                     "reasoning_token_estimate": 0, "total_token_estimate": 10,
                     "source_original_id": "", "split": "train", "source_category": "binary_only"},
    }
    with pytest.raises(ValueError, match="thinking tags"):
        module.validate_nothink_record(row, "train", 0)


def test_thinking_record_validates_binary_and_json_finals():
    module = load_module()
    estimator = module.TokenEstimator.__new__(module.TokenEstimator)
    estimator._tokenizer = None
    estimator._tried = True
    estimator._cache_dir = None

    good = {
        "id": "x", "category": "programmatic_thinking", "source": "programmatic",
        "messages": [{"role": "user", "content": "q"},
                     {"role": "assistant", "content": module.build_thinking_assistant("flip bits", "01010010")}],
        "metadata": {"thinking": True, "final_answer": "01010010",
                     "reasoning_token_estimate": 2, "total_token_estimate": 20,
                     "source_original_id": "", "split": "train", "subtype": "binary_operation_thinking"},
    }
    module.validate_thinking_record(good, "train", 0, 400, 1536, estimator)

    bad_binary = {
        "id": "x", "category": "programmatic_thinking", "source": "programmatic",
        "messages": [{"role": "user", "content": "q"},
                     {"role": "assistant", "content": module.build_thinking_assistant("r", "not8bit")}],
        "metadata": {"thinking": True, "final_answer": "not8bit",
                     "reasoning_token_estimate": 1, "total_token_estimate": 20,
                     "source_original_id": "", "split": "train", "subtype": "binary_operation_thinking"},
    }
    with pytest.raises(ValueError, match="binary final answer"):
        module.validate_thinking_record(bad_binary, "train", 0, 400, 1536, estimator)
