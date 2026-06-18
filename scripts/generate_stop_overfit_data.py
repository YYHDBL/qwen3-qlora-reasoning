"""Generate a short-answer stop overfit dataset for testing <|im_end|> learning.

The key insight: no_robots data is all long-form responses. This creates a set of
short-answer instruction-following samples (1-20 word answers) to verify that the
training pipeline can teach the model to stop correctly with the right data distribution.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

SINGLE_WORD = [
    ("Reply with exactly BLUE in uppercase and nothing else.", "BLUE"),
    ("Say only the word 'YES' in uppercase.", "YES"),
    ("Respond with just the word FALSE.", "FALSE"),
    ("Output just the word STOP.", "STOP"),
    ("Print the word HELLO and nothing else.", "HELLO"),
    ("What is 1+1? Answer with just the number.", "2"),
    ("Give me a JSON boolean for: is water wet? Output only the boolean.", "true"),
    ("Output only the word SUN.", "SUN"),
    ("Type just GREEN in uppercase.", "GREEN"),
    ("Answer with only the word MOON.", "MOON"),
]

STOP_IMMEDIATELY = [
    ("Output DONE and stop immediately. Do not explain.", "DONE"),
    ("Say OK and then stop.", "OK"),
    ("Just output FINISHED.", "FINISHED"),
    ("Reply READY and nothing more.", "READY"),
    ("Output COMPLETE and stop.", "COMPLETE"),
]

CLASSIFICATION = [
    ("Is 'dog' an animal? Answer only yes or no.", "yes"),
    ("Classify 'A square has four sides' as true or false. Output only one word.", "true"),
    ("Is Paris the capital of France? Say only yes or no.", "yes"),
    ("Classify 7 as even or odd. Output only one word.", "odd"),
    ("Is a whale a mammal? Answer only yes.", "yes"),
]

JSON_ONLY = [
    ('Return only a JSON object with key "name" and value "Alice".', '{"name":"Alice"}'),
    ('Output JSON: {"status":"ok"} and nothing else.', '{"status":"ok"}'),
    ('Emit exactly {"count":3} as JSON.', '{"count":3}'),
    ('Give me {"valid":true} in JSON only.', '{"valid":true}'),
    ('Return {"color":"blue"} as the only output.', '{"color":"blue"}'),
    ('Output only {"score":100} as JSON.', '{"score":100}'),
    ('Emit {"active":false} and nothing more.', '{"active":false}'),
]

EXTRACT = [
    ("Extract only the name from: My name is John. Output nothing else.", "John"),
    ("Extract only the email from: Reach me at admin@site.com. Output only the email.", "admin@site.com"),
    ("Extract only the number from: The answer is 42. Output just the number.", "42"),
    ("Extract only the city from: I live in Tokyo. Output only the city name.", "Tokyo"),
    ("Extract only the color from: The sky is blue. Output just the color.", "blue"),
]

REWRITE_SIMPLE = [
    ("Rewrite 'She ate the cake' in active voice. Output only the sentence.", "She ate the cake"),
    ("Rewrite 'The cat chased the mouse' in passive voice. Output only the result.", "The mouse was chased by the cat"),
    ("Make this shorter: 'I would like to request that you assist me.' Output only the short version.", "Please help me"),
]

FORMAT_CONSTRAINT = [
    ("Output exactly three uppercase ASCII letters. Nothing else.", "ABC"),
    ("Give me a valid three-letter airport code. Only the code.", "LHR"),
    ("Output exactly four digits. Nothing else.", "1234"),
    ("Return a 6-character hex color code. Only the code.", "FF00FF"),
]

SYSTEM_ADHERENCE = [
    ({"role": "system", "content": "Always answer using exactly one lowercase word."},
     "What color is grass?", "green"),
    ({"role": "system", "content": "Always answer using exactly one lowercase word."},
     "What do you call a baby dog?", "puppy"),
    ({"role": "system", "content": "Output only the requested data. No explanation, no preamble."},
     "What is the capital of Japan?", "Tokyo"),
    ({"role": "system", "content": "Respond with EXACTLY one word."},
     "What is 3+4?", "seven"),
]

FIXED_ITEM_COUNT = [
    ("List exactly three fruits, one per line, no numbering.", "apple\nbanana\norange"),
    ("List exactly two colors, one per line, no other text.", "red\ngreen"),
    ("Give exactly four letters, one per line: A, B, C, D. Output nothing else.", "A\nB\nC\nD"),
    ("Write exactly two lines: name and age. First line: Bob, second line: 25.", "Bob\n25"),
]


def _build_messages(spec):
    """Normalize message spec to list of messages."""
    if isinstance(spec, tuple) and len(spec) == 3:
        system, user, assistant = spec
        return [
            {"role": "system", "content": system["content"]},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    user, assistant = spec
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def generate_dataset(num_samples: int = 40) -> list[dict]:
    """Generate a diverse set of short-answer instruction-following samples."""
    sources = [
        ("single_word", SINGLE_WORD),
        ("stop", STOP_IMMEDIATELY),
        ("classification", CLASSIFICATION),
        ("json_only", JSON_ONLY),
        ("extract", EXTRACT),
        ("rewrite", REWRITE_SIMPLE),
        ("format", FORMAT_CONSTRAINT),
        ("system", SYSTEM_ADHERENCE),
        ("line_count", FIXED_ITEM_COUNT),
    ]

    all_samples = []
    for category, specs in sources:
        for spec in specs:
            all_samples.append({
                "category": category,
                "messages": _build_messages(spec),
            })

    random.shuffle(all_samples)
    return all_samples[:num_samples]


def main():
    out_dir = Path("data/instruction/stop_overfit")
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = generate_dataset(48)
    # Split into train (~80%) and validation (~20%)
    split = max(8, len(samples) // 5)
    train_samples = samples[split:]
    valid_samples = samples[:split]

    for split_name, data in [("train", train_samples), ("validation", valid_samples)]:
        path = out_dir / f"{split_name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for i, s in enumerate(data):
                s["id"] = f"{split_name}-{i:03d}"
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"Wrote {len(data)} samples to {path}")

    # Print a few examples
    print("\nSample examples:")
    for s in train_samples[:5]:
        msg = s["messages"]
        if len(msg) == 2:
            print(f"  [{s['category']}] User: {msg[0]['content'][:80]}")
            print(f"              Assistant: {msg[1]['content'][:80]}")
        else:
            print(f"  [{s['category']}] System: {msg[0]['content'][:60]}, User: {msg[1]['content'][:60]}")
            print(f"              Assistant: {msg[2]['content'][:80]}")
        print()


if __name__ == "__main__":
    main()
