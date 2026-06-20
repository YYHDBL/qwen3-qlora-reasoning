import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "generate_stage3_wonderland_cold_start.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stage3_wonderland", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "prompt", "answer"])
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def bit_prompt(query: str = "00001111") -> str:
    return "\n".join(
        [
            (
                "In Alice's Wonderland, a secret bit manipulation rule transforms "
                "8-bit binary numbers. The transformation involves operations like "
                "bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority or "
                "choice functions."
            ),
            "",
            "Here are some examples of input -> output:",
            "00000000 -> 11111111",
            "11111111 -> 00000000",
            "10101010 -> 01010101",
            "01010101 -> 10101010",
            "11001100 -> 00110011",
            "00110011 -> 11001100",
            "11110000 -> 00001111",
            "",
            f"Now, determine the output for: {query}",
        ]
    )


def test_reasoner_wrapper_returns_compressed_trace_and_answer():
    module = load_module()
    problem = module.parse_bit_problem("p1", bit_prompt(), "11110000")

    result = module.reason_bit_manipulation(problem)

    assert result.ok is True
    assert result.answer == "11110000"
    assert result.confidence == "high"
    assert "bit_manipulation" in result.compressed_trace
    assert "Selected 8-bit rule" in result.compressed_trace
    assert "00001111" in result.compressed_trace
    assert "\\boxed" not in result.compressed_trace
    assert len(result.compressed_trace.splitlines()) <= 8


def test_reasoner_loader_uses_repo_local_dependency():
    module = load_module()
    reasoner = module._load_bit_reasoner()

    assert reasoner.__module__ == "stage3_reasoners.bit_manipulation"


def test_generation_writes_answer_only_and_cot_with_valid_protocol(tmp_path):
    module = load_module()
    input_csv = tmp_path / "train.csv"
    output_dir = tmp_path / "stage3"
    write_csv(
        input_csv,
        [
            {"id": "00000001", "prompt": bit_prompt(), "answer": "11110000"},
            {"id": "00000002", "prompt": bit_prompt("11110000"), "answer": "00001111"},
        ],
    )

    rows, report = module.generate_stage3_dataset(
        input_csv=input_csv,
        output_dir=output_dir,
        max_prompts=2,
        seed=7,
        dry_run=False,
    )

    assert len(rows) == 4
    assert report["sample_total"] == 4
    assert report["sample_type_counts"] == {"answer_only": 2, "compressed_cot": 2}
    assert report["task_type_counts"] == {"bit_manipulation": 4}
    disk_rows = read_jsonl(output_dir / "train.jsonl")
    assert disk_rows == rows
    for row in rows:
        assert list(row["messages"][0]) == ["role", "content"]
        assert row["messages"][0]["role"] == "user"
        assert row["messages"][1]["role"] == "assistant"
        assert row["metadata"]["token_length"] <= 1024
        module.validate_stage3_record(row)
    cot_rows = [r for r in rows if r["sample_type"] == "compressed_cot"]
    assert cot_rows
    for row in cot_rows:
        assistant = row["messages"][1]["content"]
        assert assistant.count(module.THINK_OPEN) == 1
        assert assistant.count(module.THINK_CLOSE) == 1
        assert assistant.endswith(row["metadata"]["final_answer"])


def test_cot_is_skipped_when_reasoner_answer_differs_from_gold(tmp_path):
    module = load_module()
    input_csv = tmp_path / "train.csv"
    output_dir = tmp_path / "stage3"
    write_csv(input_csv, [{"id": "00000001", "prompt": bit_prompt(), "answer": "00000000"}])

    rows, report = module.generate_stage3_dataset(
        input_csv=input_csv,
        output_dir=output_dir,
        max_prompts=1,
        seed=1,
        dry_run=False,
    )

    assert [row["sample_type"] for row in rows] == ["answer_only"]
    assert report["sample_type_counts"] == {"answer_only": 1}
    assert report["skipped_reasons"]["cot_answer_mismatch"] == 1


def test_prompt_hash_deduplicates_source_prompts(tmp_path):
    module = load_module()
    input_csv = tmp_path / "train.csv"
    output_dir = tmp_path / "stage3"
    prompt = bit_prompt()
    write_csv(
        input_csv,
        [
            {"id": "00000001", "prompt": prompt, "answer": "11110000"},
            {"id": "00000002", "prompt": prompt, "answer": "11110000"},
        ],
    )

    rows, report = module.generate_stage3_dataset(
        input_csv=input_csv,
        output_dir=output_dir,
        max_prompts=10,
        seed=1,
        dry_run=False,
    )

    assert len(rows) == 2
    assert report["skipped_reasons"]["duplicate_source_prompt_hash"] == 1
    assert len({row["metadata"]["prompt_hash"] for row in rows}) == len(rows)


def test_validate_rejects_bad_cot_tail():
    module = load_module()
    row = {
        "id": "x",
        "task_type": "bit_manipulation",
        "sample_type": "compressed_cot",
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "<think>\nr\n</think>\n\n11110000\nextra"},
        ],
        "metadata": {
            "final_answer": "11110000",
            "gold_answer": "11110000",
            "token_length": 10,
            "prompt_hash": "h",
        },
    }
    with pytest.raises(ValueError, match="only final answer"):
        module.validate_stage3_record(row)
