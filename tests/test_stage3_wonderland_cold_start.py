import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "generate_stage3_wonderland_cold_start.py"


class FakeQwenTokenizer:
    name_or_path = "fake-qwen3-tokenizer"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(ch) for ch in text]


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


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_split(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"seed": 42, "stage3_sft_pool": ids}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


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


def numeral_prompt(number: str = "38") -> str:
    return "\n".join(
        [
            "In Alice's Wonderland, numbers are secretly converted into a different numeral system. Some examples are given below:",
            "11 -> XI",
            "15 -> XV",
            "94 -> XCIV",
            "19 -> XIX",
            f"Now, write the number {number} in the Wonderland numeral system.",
        ]
    )


def unit_prompt() -> str:
    return "\n".join(
        [
            "In Alice's Wonderland, a secret unit conversion is applied to measurements. For example:",
            "10.00 m becomes 20.00",
            "5.00 m becomes 10.00",
            "7.50 m becomes 15.00",
            "Now, convert the following measurement: 3.25 m",
        ]
    )


def gravity_prompt() -> str:
    return "\n".join(
        [
            "In Alice's Wonderland, the gravitational constant has been secretly changed. Here are some example observations:",
            "For t = 1.00s, distance = 2.00 m",
            "For t = 2.00s, distance = 8.00 m",
            "For t = 3.00s, distance = 18.00 m",
            "Now, determine the falling distance for t = 4.00s given d = 0.5*g*t^2.",
        ]
    )


def cipher_prompt() -> str:
    return "\n".join(
        [
            "In Alice's Wonderland, secret encryption rules are used on text. Here are some examples:",
            "bqqmf -> apple",
            "dbu -> cat",
            "cppl -> book",
            "Now, decrypt the following text: dbu cppl",
        ]
    )


def symbolic_prompt() -> str:
    return "\n".join(
        [
            "In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:",
            "12+34 = 1234",
            "56+78 = 5678",
            "90+12 = 9012",
            "Now, determine the result for: 34+56",
        ]
    )


def replay_row(row_id: str, user: str, assistant: str, category: str = "replay") -> dict:
    return {
        "id": row_id,
        "category": category,
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }


def test_task_type_detector_covers_all_wonderland_families():
    module = load_module()

    assert module.detect_task_type(bit_prompt()) == "bit_manipulation"
    assert module.detect_task_type(numeral_prompt()) == "numeral"
    assert module.detect_task_type(unit_prompt()) == "unit_conversion"
    assert module.detect_task_type(gravity_prompt()) == "gravity"
    assert module.detect_task_type(cipher_prompt()) == "cipher"
    assert module.detect_task_type(symbolic_prompt()) == "symbolic_equation"


def test_reasoner_registry_returns_uniform_result_and_debug_trace():
    module = load_module()
    registry = module.build_reasoner_registry()
    problem = module.parse_wonderland_problem(
        {"id": "n1", "prompt": numeral_prompt(), "answer": "XXXVIII"}
    )

    result = registry.run(problem)

    assert result.ok is True
    assert result.answer == "XXXVIII"
    assert result.task_type == "numeral"
    assert result.raw_trace
    assert "\\boxed" in result.raw_trace
    assert "numeral" in result.compressed_trace
    assert "\\boxed" not in result.compressed_trace


def test_generation_uses_stage3_sft_pool_only_and_writes_outputs(tmp_path):
    module = load_module()
    input_csv = tmp_path / "train.csv"
    split_path = tmp_path / "splits" / "wonderland_split_seed42.json"
    output_dir = tmp_path / "stage3"
    stage1_5 = tmp_path / "stage1_5.jsonl"
    stage2 = tmp_path / "stage2.jsonl"
    write_csv(
        input_csv,
        [
            {"id": "bit1", "prompt": bit_prompt(), "answer": "11110000"},
            {"id": "num1", "prompt": numeral_prompt(), "answer": "XXXVIII"},
            {"id": "unit1", "prompt": unit_prompt(), "answer": "6.500"},
            {"id": "leak1", "prompt": gravity_prompt(), "answer": "32.000"},
        ],
    )
    write_split(split_path, ["bit1", "num1", "unit1"])
    write_jsonl(stage1_5, [replay_row("s15-1", "Return OK only.", "OK", "exact_output")])
    write_jsonl(
        stage2,
        [
            replay_row(
                "s2-1",
                "Compute 1+1 with thinking.",
                "<think>\nAdd the two numbers.\n</think>\n\n2",
                "programmatic_thinking",
            )
        ],
    )

    rows_by_split, report = module.generate_stage3_dataset(
        input_csv=input_csv,
        split_path=split_path,
        output_dir=output_dir,
        seed=7,
        dev_ratio=0.34,
        max_stage3_prompts=3,
        stage1_5_replay_path=stage1_5,
        stage2_replay_path=stage2,
        replay_train_limit=1,
        replay_dev_limit=1,
        tokenizer=FakeQwenTokenizer(),
        dry_run=False,
    )

    assert (output_dir / "train.jsonl").exists()
    assert (output_dir / "dev.jsonl").exists()
    assert (output_dir / "report.json").exists()
    assert (output_dir / "audit.md").exists()
    assert (output_dir / "manual_review.md").exists()
    assert (output_dir / "debug" / "raw_traces.jsonl").exists()
    assert rows_by_split["train"] == read_jsonl(output_dir / "train.jsonl")
    assert rows_by_split["dev"] == read_jsonl(output_dir / "dev.jsonl")
    all_rows = rows_by_split["train"] + rows_by_split["dev"]
    assert {r["metadata"].get("source_id") for r in all_rows if r["sample_type"].startswith("wonderland_")} <= {
        "bit1",
        "num1",
        "unit1",
    }
    assert "leak1" not in {r["metadata"].get("source_id") for r in all_rows}
    assert report["source_split_leakage_check"]["ok"] is True
    assert report["sample_type_counts"]["stage1_5_strict_replay"] == 2
    assert report["sample_type_counts"]["stage2_thinking_replay"] == 2
    assert report["tokenizer"]["kind"] == "real"


def test_compressed_cot_protocol_has_only_final_answer_after_think(tmp_path):
    module = load_module()
    input_csv = tmp_path / "train.csv"
    split_path = tmp_path / "split.json"
    output_dir = tmp_path / "out"
    write_csv(input_csv, [{"id": "num1", "prompt": numeral_prompt(), "answer": "XXXVIII"}])
    write_split(split_path, ["num1"])

    rows_by_split, _ = module.generate_stage3_dataset(
        input_csv=input_csv,
        split_path=split_path,
        output_dir=output_dir,
        max_stage3_prompts=1,
        tokenizer=FakeQwenTokenizer(),
        dry_run=True,
    )

    cot_rows = [r for rows in rows_by_split.values() for r in rows if r["sample_type"] == "wonderland_compressed_cot"]
    assert cot_rows
    for row in cot_rows:
        assistant = row["messages"][1]["content"]
        parsed = module.parse_thinking(assistant)
        assert parsed is not None
        _, final = parsed
        assert final == row["metadata"]["final_answer"]
        assert "\n" not in final
        assert "<|im_end|>" not in assistant
        assert "\\boxed" not in assistant
        module.validate_stage3_record(row)


def test_cot_is_skipped_when_reasoner_answer_differs_from_gold(tmp_path):
    module = load_module()
    input_csv = tmp_path / "train.csv"
    split_path = tmp_path / "split.json"
    output_dir = tmp_path / "out"
    write_csv(input_csv, [{"id": "num1", "prompt": numeral_prompt(), "answer": "XXXVII"}])
    write_split(split_path, ["num1"])

    rows_by_split, report = module.generate_stage3_dataset(
        input_csv=input_csv,
        split_path=split_path,
        output_dir=output_dir,
        max_stage3_prompts=1,
        tokenizer=FakeQwenTokenizer(),
        dry_run=True,
    )

    sample_types = [r["sample_type"] for rows in rows_by_split.values() for r in rows]
    assert sample_types == ["wonderland_answer_only"]
    assert report["skipped_reasons"]["cot_answer_mismatch"] == 1


def test_validate_rejects_boxed_residue_and_bad_tail():
    module = load_module()
    row = {
        "id": "x",
        "task_type": "numeral",
        "sample_type": "wonderland_compressed_cot",
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "<think>\nr\n</think>\n\n\\boxed{X}"},
        ],
        "metadata": {
            "source_id": "n1",
            "source_split": "stage3_sft_pool",
            "final_answer": "X",
            "gold_answer": "X",
            "token_length": 10,
            "prompt_hash": "h",
            "reasoner": {"ok": True, "answer": "X"},
        },
    }

    with pytest.raises(ValueError, match="boxed"):
        module.validate_stage3_record(row)

    row["messages"][1]["content"] = "<think>\nr\n</think>\n\nX\nextra"
    with pytest.raises(ValueError, match="only final answer"):
        module.validate_stage3_record(row)


def test_missing_split_file_is_a_hard_error(tmp_path):
    module = load_module()
    input_csv = tmp_path / "train.csv"
    write_csv(input_csv, [{"id": "num1", "prompt": numeral_prompt(), "answer": "XXXVIII"}])

    with pytest.raises(FileNotFoundError, match="wonderland split"):
        module.generate_stage3_dataset(
            input_csv=input_csv,
            split_path=tmp_path / "missing.json",
            output_dir=tmp_path / "out",
            tokenizer=FakeQwenTokenizer(),
            dry_run=True,
        )
