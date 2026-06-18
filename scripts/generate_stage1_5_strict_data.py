#!/usr/bin/env python3
"""Generate Stage 1.5 strict-format / stop-behavior instruction data.

This script only synthesizes small rule-based strict-format samples and replays
filtered Stage 1 No Robots records. It does not use Wonderland data or external
datasets.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


TRAIN_COUNTS = {
    "exact_output": 200,
    "binary_only": 250,
    "wonderland_like_binary": 250,
    "json_only": 200,
    "yes_no": 200,
    "classification": 200,
    "extraction": 120,
    "line_count": 100,
    "stop_behavior": 80,
    "no_robots_replay": 400,
}
VALIDATION_COUNTS = {
    "exact_output": 20,
    "binary_only": 25,
    "wonderland_like_binary": 25,
    "json_only": 20,
    "yes_no": 20,
    "classification": 20,
    "extraction": 12,
    "line_count": 10,
    "stop_behavior": 8,
    "no_robots_replay": 40,
}

STRICT_CATEGORIES = set(TRAIN_COUNTS) - {"no_robots_replay"}
PREFERRED_REPLAY_CATEGORIES = {
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
STRICT_BAD_PREFIXES = (
    "sure",
    "here is",
    "the answer is",
    "i hope this helps",
    "```",
)

EXACT_OUTPUTS = [
    "BLUE",
    "GREEN",
    "RED",
    "DONE",
    "OK",
    "YES",
    "NO",
    "PASS",
    "FAIL",
    "READY",
    "STOP",
]
EXACT_TEMPLATES = [
    "Reply with exactly {answer} and nothing else.",
    "Output only {answer}. Do not explain.",
    "Return exactly {answer}.",
    "Your entire answer must be {answer}.",
    "Say {answer} and stop. No other text.",
    "Respond with the single token {answer}.",
    "Print exactly {answer}, with no prefix or suffix.",
    "The only valid response is {answer}.",
]
BINARY_TEMPLATES = [
    "Return only this 8-bit binary string and nothing else: {answer}.",
    "Output exactly this 8-bit binary answer: {answer}. Do not explain.",
    "Your answer must be exactly one 8-bit binary string: {answer}.",
    "Copy this binary answer exactly and stop: {answer}.",
    "Reply with {answer} only. No words, no punctuation.",
    "Emit the following 8-bit binary string as the full answer: {answer}.",
]
STOP_TEMPLATES = [
    "Output {answer} and stop immediately.",
    "Reply with {answer}. Do not write anything else.",
    "Say only {answer}, then stop.",
    "Return {answer} as the complete response.",
    "Print {answer}. No explanation, no second line.",
]


def eight_bit(rng: random.Random) -> str:
    return format(rng.randrange(256), "08b")


def rotate_left(bits: str, amount: int) -> str:
    return bits[amount:] + bits[:amount]


def rotate_right(bits: str, amount: int) -> str:
    return bits[-amount:] + bits[:-amount]


def bitwise_binary_op(bits: str, mask: str, op: Callable[[int, int], int]) -> str:
    return "".join(str(op(int(bit), int(mask_bit))) for bit, mask_bit in zip(bits, mask))


def build_wonderland_rule(rng: random.Random) -> tuple[str, Callable[[str], str]]:
    rule_name = rng.choice(
        [
            "not",
            "rotate_left",
            "rotate_right",
            "xor",
            "and",
            "or",
            "shift_left",
            "shift_right",
        ]
    )
    if rule_name == "not":
        return "bitwise_not", lambda bits: "".join("1" if bit == "0" else "0" for bit in bits)
    if rule_name == "rotate_left":
        amount = rng.randint(1, 3)
        return f"rotate_left_{amount}", lambda bits: rotate_left(bits, amount)
    if rule_name == "rotate_right":
        amount = rng.randint(1, 3)
        return f"rotate_right_{amount}", lambda bits: rotate_right(bits, amount)
    if rule_name == "xor":
        mask = eight_bit(rng)
        return f"xor_{mask}", lambda bits: bitwise_binary_op(bits, mask, lambda a, b: a ^ b)
    if rule_name == "and":
        mask = eight_bit(rng)
        return f"and_{mask}", lambda bits: bitwise_binary_op(bits, mask, lambda a, b: a & b)
    if rule_name == "or":
        mask = eight_bit(rng)
        return f"or_{mask}", lambda bits: bitwise_binary_op(bits, mask, lambda a, b: a | b)
    if rule_name == "shift_left":
        amount = rng.randint(1, 3)
        return f"shift_left_{amount}", lambda bits: bits[amount:] + ("0" * amount)
    amount = rng.randint(1, 3)
    return f"shift_right_{amount}", lambda bits: ("0" * amount) + bits[:-amount]


def compact_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def make_messages(user: str, assistant: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def make_record(
    split: str,
    category: str,
    index: int,
    user: str,
    assistant: str,
    validator: dict[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "id": f"stage1_5-{split}-{category}-{index:05d}",
        "category": category,
        "messages": make_messages(user, assistant),
        "validator": validator,
    }
    if extra:
        record.update(extra)
    return record


def generate_exact_output(split: str, count: int, rng: random.Random) -> list[dict[str, Any]]:
    rows = []
    for index in range(count):
        answer = rng.choice(EXACT_OUTPUTS)
        user = rng.choice(EXACT_TEMPLATES).format(answer=answer)
        rows.append(
            make_record(
                split,
                "exact_output",
                index,
                user,
                answer,
                {"type": "exact", "value": answer},
            )
        )
    return rows


def generate_binary_only(split: str, count: int, rng: random.Random) -> list[dict[str, Any]]:
    rows = []
    for index in range(count):
        answer = eight_bit(rng)
        user = rng.choice(BINARY_TEMPLATES).format(answer=answer)
        rows.append(
            make_record(
                split,
                "binary_only",
                index,
                user,
                answer,
                {"type": "regex", "pattern": "^[01]{8}$"},
            )
        )
    return rows


def generate_wonderland_like_binary(
    split: str, count: int, rng: random.Random
) -> list[dict[str, Any]]:
    rows = []
    for index in range(count):
        rule_name, transform = build_wonderland_rule(rng)
        inputs = set()
        while len(inputs) < 5:
            inputs.add(eight_bit(rng))
        values = list(inputs)
        examples = values[:4]
        query = values[4]
        answer = transform(query)
        example_lines = "\n".join(f"{bits} -> {transform(bits)}" for bits in examples)
        user = (
            "Given the following input-output examples for one consistent 8-bit rule:\n"
            f"{example_lines}\n"
            f"Now determine the output for: {query}\n"
            "Return only the final 8-bit binary answer. Do not explain."
        )
        rows.append(
            make_record(
                split,
                "wonderland_like_binary",
                index,
                user,
                answer,
                {"type": "regex", "pattern": "^[01]{8}$"},
                {"rule": rule_name},
            )
        )
    return rows


def generate_json_only(split: str, count: int, rng: random.Random) -> list[dict[str, Any]]:
    templates = [
        'Return only valid JSON with key "{key}" and value {value}. Do not include markdown or explanation.',
        'Output a compact JSON object: key "{key}", value {value}. Return JSON only.',
        'Respond only with valid JSON containing "{key}": {value}.',
        'No prose. No markdown. Return exactly the JSON object with "{key}" set to {value}.',
    ]
    rows = []
    for index in range(count):
        kind = rng.choice(["sum", "product", "answer", "label", "binary"])
        if kind == "sum":
            a, b = rng.randint(0, 20), rng.randint(0, 20)
            value: Any = a + b
            user_value = str(value)
            user = rng.choice(templates).format(key="sum", value=user_value)
            assistant = compact_json({"sum": value})
        elif kind == "product":
            a, b = rng.randint(0, 12), rng.randint(0, 12)
            value = a * b
            user = rng.choice(templates).format(key="product", value=str(value))
            assistant = compact_json({"product": value})
        elif kind == "answer":
            value = rng.choice(["yes", "no"])
            user = rng.choice(templates).format(key="answer", value=json.dumps(value))
            assistant = compact_json({"answer": value})
        elif kind == "label":
            value = rng.choice(sorted(CLASSIFICATION_LABELS))
            user = rng.choice(templates).format(key="label", value=json.dumps(value))
            assistant = compact_json({"label": value})
        else:
            value = eight_bit(rng)
            user = rng.choice(templates).format(key="binary", value=json.dumps(value))
            assistant = compact_json({"binary": value})
        rows.append(
            make_record(
                split,
                "json_only",
                index,
                user,
                assistant,
                {"type": "json"},
            )
        )
    return rows


def generate_yes_no(split: str, count: int, rng: random.Random) -> list[dict[str, Any]]:
    rows = []
    for index in range(count):
        kind = rng.choice(["greater", "less", "equal", "even", "divisible"])
        if kind == "greater":
            a, b = rng.randint(0, 30), rng.randint(0, 30)
            question = f"Is {a} greater than {b}?"
            answer = "yes" if a > b else "no"
        elif kind == "less":
            a, b = rng.randint(0, 30), rng.randint(0, 30)
            question = f"Is {a} less than {b}?"
            answer = "yes" if a < b else "no"
        elif kind == "equal":
            a = rng.randint(0, 20)
            b = a if rng.random() < 0.5 else rng.randint(0, 20)
            question = f"Is {a} equal to {b}?"
            answer = "yes" if a == b else "no"
        elif kind == "even":
            a = rng.randint(0, 60)
            question = f"Is {a} an even number?"
            answer = "yes" if a % 2 == 0 else "no"
        else:
            a = rng.randint(1, 80)
            b = rng.randint(2, 10)
            question = f"Is {a} divisible by {b}?"
            answer = "yes" if a % b == 0 else "no"
        user = f"{question} Answer only yes or no. Do not explain."
        rows.append(
            make_record(
                split,
                "yes_no",
                index,
                user,
                answer,
                {"type": "exact", "value": answer},
            )
        )
    return rows


def generate_classification(split: str, count: int, rng: random.Random) -> list[dict[str, Any]]:
    examples = {
        "positive": [
            "I love this product.",
            "The service was fast and friendly.",
            "This update works beautifully.",
            "I am happy with the result.",
        ],
        "negative": [
            "I hate how slow this app is.",
            "The package arrived broken.",
            "This was a frustrating experience.",
            "The result is disappointing.",
        ],
        "neutral": [
            "The meeting starts at noon.",
            "The box contains three cables.",
            "The report was published today.",
            "The device weighs two kilograms.",
        ],
    }
    rows = []
    for index in range(count):
        label = rng.choice(sorted(examples))
        text = rng.choice(examples[label])
        user = (
            "Classify the sentiment as positive, negative, or neutral. "
            "Output only the label.\n"
            f"Text: {text}"
        )
        rows.append(
            make_record(
                split,
                "classification",
                index,
                user,
                label,
                {"type": "exact", "value": label, "choices": sorted(CLASSIFICATION_LABELS)},
            )
        )
    return rows


def generate_extraction(split: str, count: int, rng: random.Random) -> list[dict[str, Any]]:
    first_names = ["Mira", "Jon", "Ava", "Nolan", "Priya", "Eli", "Rosa", "Theo"]
    domains = ["example.com", "mail.test", "sample.org", "inbox.dev"]
    rows = []
    for index in range(count):
        kind = rng.choice(["email", "phone", "name", "date", "order"])
        if kind == "email":
            value = f"{rng.choice(first_names).lower()}{rng.randint(10, 99)}@{rng.choice(domains)}"
            user = f"Extract the email address. Output only the extracted text.\nText: Please contact {value} before Friday."
        elif kind == "phone":
            value = f"555-{rng.randint(100, 999)}-{rng.randint(1000, 9999)}"
            user = f"Extract the phone number. Output only the extracted text.\nText: Call the support desk at {value} after 3 PM."
        elif kind == "name":
            value = f"{rng.choice(first_names)} {rng.choice(['Chen', 'Patel', 'Rivera', 'Smith', 'Kim'])}"
            user = f"Extract the full name. Output only the extracted text.\nText: The account owner is {value}, created yesterday."
        elif kind == "date":
            value = f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
            user = f"Extract the date. Output only the extracted text.\nText: The maintenance window is scheduled for {value} at night."
        else:
            value = f"ORD-{rng.randint(100000, 999999)}"
            user = f"Extract the order ID. Output only the extracted text.\nText: Refund request received for order {value}."
        rows.append(
            make_record(
                split,
                "extraction",
                index,
                user,
                value,
                {"type": "exact", "value": value},
            )
        )
    return rows


def generate_line_count(split: str, count: int, rng: random.Random) -> list[dict[str, Any]]:
    pools = [
        ("two colors", 2, ["Red", "Blue", "Green", "Yellow", "Black", "White"]),
        ("three fruit names", 3, ["Apple", "Banana", "Orange", "Mango", "Pear", "Grape"]),
        ("four animal names", 4, ["Cat", "Dog", "Horse", "Panda", "Tiger", "Zebra"]),
        ("three city names", 3, ["Paris", "Tokyo", "Berlin", "Lima", "Seoul", "Cairo"]),
    ]
    rows = []
    for index in range(count):
        label, line_count, values = rng.choice(pools)
        selected = rng.sample(values, line_count)
        assistant = "\n".join(selected)
        user = f"Output exactly {label}, one per line. Do not add anything else."
        rows.append(
            make_record(
                split,
                "line_count",
                index,
                user,
                assistant,
                {"type": "line_count", "count": line_count},
            )
        )
    return rows


def generate_stop_behavior(split: str, count: int, rng: random.Random) -> list[dict[str, Any]]:
    rows = []
    for index in range(count):
        answer = rng.choice(["DONE", "OK", "STOP", "READY", "PASS"])
        user = rng.choice(STOP_TEMPLATES).format(answer=answer)
        rows.append(
            make_record(
                split,
                "stop_behavior",
                index,
                user,
                answer,
                {"type": "exact", "value": answer},
            )
        )
    return rows


GENERATORS: dict[str, Callable[[str, int, random.Random], list[dict[str, Any]]]] = {
    "exact_output": generate_exact_output,
    "binary_only": generate_binary_only,
    "wonderland_like_binary": generate_wonderland_like_binary,
    "json_only": generate_json_only,
    "yes_no": generate_yes_no,
    "classification": generate_classification,
    "extraction": generate_extraction,
    "line_count": generate_line_count,
    "stop_behavior": generate_stop_behavior,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
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


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def get_user_and_assistant(messages: Any) -> tuple[str, str]:
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    user_messages = [
        message.get("content")
        for message in messages
        if isinstance(message, Mapping) and message.get("role") == "user"
    ]
    assistant_messages = [
        message.get("content")
        for message in messages
        if isinstance(message, Mapping) and message.get("role") == "assistant"
    ]
    if not user_messages or not assistant_messages:
        raise ValueError("messages must contain user and assistant messages")
    user = user_messages[0]
    assistant = assistant_messages[-1]
    if not isinstance(user, str) or not isinstance(assistant, str):
        raise ValueError("message content must be strings")
    return user, assistant


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def looks_clean_text(text: str) -> bool:
    if "\x00" in text or "\ufffd" in text:
        return False
    if text and sum(ch.isprintable() or ch.isspace() for ch in text) / len(text) < 0.98:
        return False
    stripped = text.strip()
    return bool(stripped) and len(set(stripped)) > 8


def replay_candidate(row: Mapping[str, Any]) -> bool:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return False
    try:
        user, assistant = get_user_and_assistant(messages)
    except ValueError:
        return False
    if len(user) >= 2500:
        return False
    count = word_count(assistant)
    if count < 50 or count > 250:
        return False
    lower = assistant.lstrip().lower()
    if lower.startswith(("i'm sorry", "i’m sorry", "i cannot")):
        return False
    if row.get("category") not in PREFERRED_REPLAY_CATEGORIES:
        return False
    return looks_clean_text(user) and looks_clean_text(assistant)


def select_no_robots_replay(
    path: Path, count: int, rng: random.Random, split: str
) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    candidates = [row for row in rows if replay_candidate(row)]
    if len(candidates) < count:
        raise ValueError(
            f"{split} No Robots replay candidates insufficient: "
            f"need {count}, found {len(candidates)} from {path}"
        )
    selected = rng.sample(candidates, count)
    replay_rows = []
    for index, row in enumerate(selected):
        source_id = str(row.get("id") or row.get("prompt_id") or f"{split}-{index}")
        replay = {
            "id": f"stage1_5-{split}-no_robots_replay-{index:05d}",
            "category": "no_robots_replay",
            "messages": copy.deepcopy(row["messages"]),
            "source_id": source_id,
            "source_category": row.get("category"),
            "source": row.get("source", "HuggingFaceH4/no_robots"),
        }
        for field in ("source_split", "source_row_index", "prompt_id"):
            if field in row:
                replay[field] = row[field]
        replay_rows.append(replay)
    return replay_rows


def generate_split(
    split: str,
    counts: Mapping[str, int],
    replay_path: Path,
    rng: random.Random,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for category, count in counts.items():
        if category == "no_robots_replay":
            rows.extend(select_no_robots_replay(replay_path, count, rng, split))
        else:
            rows.extend(GENERATORS[category](split, count, rng))
    rng.shuffle(rows)
    return rows


def validate_record(row: Mapping[str, Any], split: str, index: int) -> None:
    location = f"{split}[{index}]"
    record_id = row.get("id")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError(f"{location} missing nonempty id")
    category = row.get("category")
    if not isinstance(category, str) or category not in TRAIN_COUNTS:
        raise ValueError(f"{location} unsupported category: {category!r}")
    messages = row.get("messages")
    user, assistant = get_user_and_assistant(messages)
    if not user:
        raise ValueError(f"{location} user content is empty")
    if not assistant:
        raise ValueError(f"{location} assistant content is empty")
    if "<|im_end|>" in assistant:
        raise ValueError(f"{location} assistant content must not contain <|im_end|>")

    if category in STRICT_CATEGORIES:
        stripped = assistant.strip()
        if assistant != stripped:
            raise ValueError(f"{location} strict assistant has surrounding whitespace")
        lowered = stripped.lower()
        if any(lowered.startswith(prefix) for prefix in STRICT_BAD_PREFIXES):
            raise ValueError(f"{location} strict assistant has explanatory prefix")
        if "```" in stripped:
            raise ValueError(f"{location} strict assistant contains markdown fence")

    if category in {"binary_only", "wonderland_like_binary"}:
        if not re.fullmatch(r"[01]{8}", assistant):
            raise ValueError(f"{location} assistant must be one 8-bit binary string")
    elif category == "json_only":
        try:
            json.loads(assistant)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{location} assistant must be valid JSON") from exc
    elif category == "yes_no":
        if assistant not in {"yes", "no"}:
            raise ValueError(f"{location} assistant must be yes or no")
    elif category == "classification":
        if assistant not in CLASSIFICATION_LABELS:
            raise ValueError(f"{location} assistant must be a classification label")
    elif category == "line_count":
        validator = row.get("validator")
        expected = validator.get("count") if isinstance(validator, Mapping) else None
        if not isinstance(expected, int):
            raise ValueError(f"{location} line_count record missing validator count")
        if len(assistant.splitlines()) != expected:
            raise ValueError(f"{location} assistant has wrong line count")


def validate_dataset(rows: list[dict[str, Any]], split: str) -> None:
    ids = set()
    for index, row in enumerate(rows):
        validate_record(row, split, index)
        record_id = row["id"]
        if record_id in ids:
            raise ValueError(f"{split}[{index}] duplicate id: {record_id}")
        ids.add(record_id)


def length_stats(values: list[int]) -> dict[str, float]:
    return {
        "min": min(values),
        "max": max(values),
        "mean": round(statistics.fmean(values), 2),
    }


def split_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    user_lengths = []
    assistant_lengths = []
    for row in rows:
        user, assistant = get_user_and_assistant(row["messages"])
        user_lengths.append(len(user))
        assistant_lengths.append(len(assistant))
    return {
        "count": len(rows),
        "categories": dict(Counter(row["category"] for row in rows)),
        "user_chars": length_stats(user_lengths),
        "assistant_chars": length_stats(assistant_lengths),
        "no_robots_replay": sum(row["category"] == "no_robots_replay" for row in rows),
    }


def print_stats(split: str, rows: list[dict[str, Any]]) -> None:
    stats = split_stats(rows)
    print(f"{split} sample count: {stats['count']}")
    print(f"{split} category counts:")
    for category in TRAIN_COUNTS:
        print(f"  {category}: {stats['categories'].get(category, 0)}")
    print(f"{split} user length chars: {stats['user_chars']}")
    print(f"{split} assistant length chars: {stats['assistant_chars']}")
    print(f"{split} No Robots replay selected: {stats['no_robots_replay']}")


def generate_stage1_5_dataset(
    seed: int,
    output_dir: Path,
    stage1_train: Path,
    stage1_validation: Path,
    train_counts: Mapping[str, int] = TRAIN_COUNTS,
    validation_counts: Mapping[str, int] = VALIDATION_COUNTS,
) -> dict[str, Path]:
    train_rng = random.Random(seed)
    validation_rng = random.Random(seed + 10_000)
    train_rows = generate_split("train", train_counts, stage1_train, train_rng)
    validation_rows = generate_split(
        "validation", validation_counts, stage1_validation, validation_rng
    )

    validate_dataset(train_rows, "train")
    validate_dataset(validation_rows, "validation")
    overlap = {row["id"] for row in train_rows} & {row["id"] for row in validation_rows}
    if overlap:
        raise ValueError(f"train and validation IDs overlap: {sorted(overlap)[:5]}")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(validation_path, validation_rows)

    print_stats("train", train_rows)
    print_stats("validation", validation_rows)
    print(f"train output path: {train_path}")
    print(f"validation output path: {validation_path}")
    return {"train": train_path, "validation": validation_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Stage 1.5 strict-format / stop-behavior JSONL data."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("data/instruction/stage1_5"))
    parser.add_argument(
        "--stage1-train",
        type=Path,
        default=Path("data/instruction/stage1/train.jsonl"),
    )
    parser.add_argument(
        "--stage1-validation",
        type=Path,
        default=Path("data/instruction/stage1/validation.jsonl"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_stage1_5_dataset(
        seed=args.seed,
        output_dir=args.output_dir,
        stage1_train=args.stage1_train,
        stage1_validation=args.stage1_validation,
    )


if __name__ == "__main__":
    main()
