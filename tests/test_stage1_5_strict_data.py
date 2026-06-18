import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "generate_stage1_5_strict_data.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stage1_5_strict_data", SCRIPT_PATH)
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
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def replay_row(index: int, *, category: str = "Open QA", prefix: str = "Answer") -> dict:
    words = " ".join(f"{prefix}{index}_{word}" for word in range(70))
    return {
        "id": f"stage1-{index}",
        "category": category,
        "messages": [
            {"role": "user", "content": f"Question {index}?"},
            {"role": "assistant", "content": words},
        ],
        "source": "HuggingFaceH4/no_robots",
        "source_split": "train",
        "source_row_index": index,
    }


def test_stage1_5_generation_writes_expected_counts_and_valid_records(tmp_path):
    module = load_module()
    train_path = tmp_path / "stage1" / "train.jsonl"
    validation_path = tmp_path / "stage1" / "validation.jsonl"
    write_jsonl(train_path, [replay_row(i) for i in range(5)])
    write_jsonl(validation_path, [replay_row(i, prefix="Valid") for i in range(3)])

    train_counts = {
        "exact_output": 2,
        "binary_only": 2,
        "wonderland_like_binary": 2,
        "json_only": 2,
        "yes_no": 2,
        "classification": 2,
        "extraction": 2,
        "line_count": 2,
        "stop_behavior": 2,
        "no_robots_replay": 3,
    }
    validation_counts = {category: 1 for category in train_counts}
    output_dir = tmp_path / "stage1_5"

    module.generate_stage1_5_dataset(
        seed=123,
        output_dir=output_dir,
        stage1_train=train_path,
        stage1_validation=validation_path,
        train_counts=train_counts,
        validation_counts=validation_counts,
    )

    train_rows = read_jsonl(output_dir / "train.jsonl")
    validation_rows = read_jsonl(output_dir / "validation.jsonl")

    assert Counter(row["category"] for row in train_rows) == train_counts
    assert Counter(row["category"] for row in validation_rows) == validation_counts
    assert len({row["id"] for row in train_rows + validation_rows}) == (
        len(train_rows) + len(validation_rows)
    )
    module.validate_dataset(train_rows, "train")
    module.validate_dataset(validation_rows, "validation")


def test_no_robots_replay_requires_enough_filtered_candidates(tmp_path):
    module = load_module()
    path = tmp_path / "train.jsonl"
    write_jsonl(path, [replay_row(0, category="Other")])

    with pytest.raises(ValueError, match="No Robots replay candidates"):
        module.select_no_robots_replay(path, count=1, rng=module.random.Random(1), split="train")
