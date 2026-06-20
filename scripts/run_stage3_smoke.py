#!/usr/bin/env python3
"""Stage 3 Wonderland Cold Start - smoke training + evaluation.

Phases:
  1. Freeze: sha256 manifest of current Stage 3 data
  2. Label audit: annotate all stage3 training data with assistant-only labels
  3. Smoke sampling: 300 records stratified by sample_type x task_type
  4. Training: continue from Stage 2 adapter, smoke mode
  5. Evaluation: adapter sanity, Wonderland dev, strict/thinking/open regression
  6. Reports: smoke_train_report.json + smoke_eval_report.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.config import load_yaml_config, apply_overrides
from src.common.experiment import sha256_file, write_json, write_jsonl

STAGE3_DATA_DIR = Path("data/instruction/stage3_wonderland_cold_start")
SMOKE_DATA_DIR = Path("data/instruction/stage3_wonderland_cold_start/smoke_300")
STAGE2_ADAPTER = Path("outputs/stage2_thinking_warmup/formal/adapter")
SMOKE_OUTPUT_DIR = Path("outputs/stage3_wonderland_coldstart/smoke_300")

SAMPLE_TYPE_LIMITS = {
    "wonderland_answer_only": 120,
    "wonderland_compressed_cot": 120,
    "stage1_5_strict_replay": 30,
    "stage2_thinking_replay": 30,
}

# ── Phase 1: Data freeze ─────────────────────────────────────────────────────


def phase1_freeze() -> dict[str, Any]:
    """Record sha256 of all Stage 3 data artifacts."""
    print("[phase1] Freezing Stage 3 data versions...")

    files = {}
    for fname in ["train.jsonl", "dev.jsonl", "report.json"]:
        fpath = STAGE3_DATA_DIR / fname
        if fpath.is_file():
            files[fname] = {
                "path": str(fpath),
                "sha256": sha256_file(fpath),
                "size_bytes": fpath.stat().st_size,
            }

    manifest = {
        "data_dir": str(STAGE3_DATA_DIR),
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": files,
    }

    manifest_path = SMOKE_DATA_DIR / "dataset_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    # Also save split manifest separately for training artifact snapshot
    write_json(
        STAGE3_DATA_DIR / "dataset_manifest.json",
        {
            "train": files.get("train.jsonl", {}) if "train.jsonl" in files else {},
            "validation": files.get("dev.jsonl", {}) if "dev.jsonl" in files else {},
        },
    )

    print(f"  Manifest written to {manifest_path}")
    return manifest


# ── Phase 2: Label audit ─────────────────────────────────────────────────────


def phase2_label_audit(config: dict[str, Any]) -> dict[str, Any]:
    """Run label audit on the smoke data directory."""
    print("[phase2] Running label audit on smoke dataset...")

    from src.training.label_audit import run_label_audit

    run_label_audit(config)

    # Load results for verification
    token_report = json.loads(
        (SMOKE_DATA_DIR / "token_report.json").read_text(encoding="utf-8")
    )
    batch_audit = json.loads(
        (SMOKE_DATA_DIR / "batch_audit.json").read_text(encoding="utf-8")
    )

    train_split = token_report["splits"]["train"]
    val_split = token_report["splits"]["validation"]

    print(
        f"  train: {train_split['count']} total, "
        f"{train_split['eligible_count']} eligible, "
        f"{train_split['excluded_count']} excluded"
    )
    print(
        f"  validation: {val_split['count']} total, "
        f"{val_split['eligible_count']} eligible, "
        f"{val_split['excluded_count']} excluded"
    )
    print(f"  batch_audit status: {batch_audit['status']}")

    checks = batch_audit.get("checks", {})
    for check_name, passed in checks.items():
        status_str = "PASS" if passed else "FAIL"
        print(f"    {check_name}: {status_str}")

    if batch_audit["status"] != "passed":
        raise RuntimeError("Label audit failed - fix issues before training")

    return {"token_report": token_report, "batch_audit": batch_audit}


# ── Phase 3: Stratified smoke sampling ───────────────────────────────────────


def _largest_remainder_allocation(total: int, proportions: Sequence[tuple[str, int]]) -> dict[str, int]:
    """Allocate N items across groups proportional to counts using Hare quota."""
    total_count = sum(cnt for _, cnt in proportions)
    if total_count == 0:
        return {}

    quota = total / total_count
    allocated: dict[str, int] = {}
    remainders: list[tuple[float, str, int]] = []

    assigned_sum = 0
    for key, cnt in proportions:
        raw = cnt * quota
        alloc = int(raw)
        allocated[key] = alloc
        assigned_sum += alloc
        remainders.append((raw - alloc, key, cnt))

    # Assign remaining slots by largest remainder
    remainders.sort(key=lambda x: x[0], reverse=True)
    for i in range(total - assigned_sum):
        if i >= len(remainders):
            break
        allocated[remainders[i][1]] += 1

    return allocated


def phase3_sample() -> list[dict[str, Any]]:
    """Sample 300 records from train.jsonl, stratified by sample_type and task_type."""
    print("[phase3] Sampling 300 smoke records...")

    random.seed(42)

    # Load all train records
    records: list[dict[str, Any]] = []
    with open(STAGE3_DATA_DIR / "train.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"  Loaded {len(records)} train records")

    # Group by (sample_type, task_type)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        key = (r["sample_type"], r["task_type"])
        groups[key].append(r)

    # Shuffle each group deterministically
    for group_records in groups.values():
        random.shuffle(group_records)

    # Compute allocation
    sampled: list[dict] = []
    for sample_type, limit in SAMPLE_TYPE_LIMITS.items():
        # Find all task_type groups for this sample_type
        st_groups = {
            task_type: len(group_records)
            for (st, task_type), group_records in groups.items()
            if st == sample_type
        }
        if not st_groups:
            print(f"  WARNING: no records for sample_type={sample_type}")
            continue

        # Proportional allocation across task_types
        proportions = list(st_groups.items())
        alloc = _largest_remainder_allocation(limit, proportions)

        # Take allocated count from each shuffled group
        for task_type, take_n in alloc.items():
            group_records = groups.get((sample_type, task_type), [])
            if take_n > len(group_records):
                print(
                    f"  WARNING: {sample_type}/{task_type} needs {take_n} "
                    f"but only {len(group_records)} available"
                )
                take_n = len(group_records)
            sampled.extend(group_records[:take_n])
            print(
                f"    {sample_type}/{task_type}: {take_n}/{len(group_records)}"
            )

    # Shuffle the final 300
    random.shuffle(sampled)
    print(f"  Total sampled: {len(sampled)}")

    if len(sampled) != 300:
        print(f"  WARNING: expected 300 samples, got {len(sampled)}")

    return sampled


def phase3_write_smoke_data(sampled: list[dict[str, Any]]) -> None:
    """Write sampled train data + copy dev as validation."""
    SMOKE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    write_jsonl(SMOKE_DATA_DIR / "train.jsonl", sampled)

    # Copy dev.jsonl as validation.jsonl (for smoke eval)
    dev_path = STAGE3_DATA_DIR / "dev.jsonl"
    import shutil
    shutil.copy2(dev_path, SMOKE_DATA_DIR / "validation.jsonl")

    print(f"  Wrote {len(sampled)} to {SMOKE_DATA_DIR / 'train.jsonl'}")
    print(f"  Copied dev → {SMOKE_DATA_DIR / 'validation.jsonl'}")

    # Compute sha256 of resulting files
    for fname in ["train.jsonl", "validation.jsonl"]:
        fpath = SMOKE_DATA_DIR / fname
        if fpath.is_file():
            print(f"    {fname}: sha256={sha256_file(fpath)}")


# ── Phase 4: Training ────────────────────────────────────────────────────────


def phase4_train(config: dict[str, Any]) -> dict[str, Any]:
    """Run smoke training from Stage 2 adapter."""
    print("[phase4] Starting smoke training...")

    # Override output_root for smoke subdirectory
    config = dict(config)
    config["experiment"] = dict(config["experiment"])
    config["experiment"]["output_root"] = str(SMOKE_OUTPUT_DIR)

    from src.training.train_sft import run_training as sft_run_training

    metrics = sft_run_training(
        config,
        run_mode="smoke",
        adapter_path=STAGE2_ADAPTER,
    )

    print(f"  Training complete. Steps: {metrics.get('global_step')}")
    return metrics


# ── Phase 5: Evaluation ──────────────────────────────────────────────────────


def _load_model_for_eval(config: dict[str, Any], adapter_path: str):
    """Load model + tokenizer + stop_ids for evaluation."""
    import torch
    from src.training.model_loader import load_lora_model, load_tokenizer
    from src.training.chat_template import resolve_stop_token_ids

    tokenizer = load_tokenizer(config, for_training=False)
    model = load_lora_model(config, adapter_path, is_trainable=False).eval()
    model.config.use_cache = True
    stop_ids = resolve_stop_token_ids(tokenizer)
    return model, tokenizer, stop_ids


def _generate(
    model: Any,
    tokenizer: Any,
    stop_ids: dict[str, int],
    messages: list[dict],
    max_new_tokens: int = 256,
    enable_thinking: bool | None = None,
) -> str:
    """Generate text from messages list."""
    import torch
    from src.training.chat_template import render_generation_prompt

    prompt_text = render_generation_prompt(tokenizer, list(messages), enable_thinking=enable_thinking)
    inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=[tokenizer.eos_token_id, stop_ids.get("im_end"), stop_ids.get("endoftext")],
        )
    generated_ids = outputs[0][input_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def phase5_eval(config: dict[str, Any]) -> dict[str, Any]:
    """Comprehensive evaluation suite."""
    print("[phase5] Running evaluation suite...")

    adapter_path = str(SMOKE_OUTPUT_DIR / "smoke" / "adapter")

    results: dict[str, Any] = {}

    # 5a: Adapter reload sanity
    print("  [5a] Adapter reload sanity...")
    results["adapter_sanity"] = phase5a_adapter_sanity(config, adapter_path)

    # 5b: Wonderland dev subset
    print("  [5b] Wonderland dev eval...")
    results["wonderland_dev"] = phase5b_wonderland_dev(config, adapter_path)

    # 5c: Stage 1.5 strict regression
    print("  [5c] Stage 1.5 strict regression...")
    results["stage1_5_strict"] = phase5c_strict_regression(config, adapter_path)

    # 5d: Stage 2 protocol regression
    print("  [5d] Stage 2 protocol regression...")
    results["stage2_protocol"] = phase5d_protocol_regression(config, adapter_path)

    # 5e: Open regression
    print("  [5e] Open replay regression...")
    results["open_regression"] = phase5e_open_regression(config, adapter_path)

    return results


def phase5a_adapter_sanity(config: dict[str, Any], adapter_path: str) -> dict[str, Any]:
    """Verify adapter loads and generates properly."""
    import torch

    model, tokenizer, stop_ids = _load_model_for_eval(config, adapter_path)

    # Quick generation test with a simple prompt
    test_prompt = [{"role": "user", "content": "What is 2+2? Answer in one word."}]
    start = time.time()
    generated = _generate(model, tokenizer, stop_ids, test_prompt, max_new_tokens=32, enable_thinking=False)
    elapsed = time.time() - start

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "passed": len(generated) > 0 and len(generated) < 200,
        "test_prompt": "What is 2+2? Answer in one word.",
        "generated": generated,
        "generated_length": len(generated),
        "elapsed_sec": round(elapsed, 2),
    }


def phase5b_wonderland_dev(config: dict[str, Any], adapter_path: str) -> dict[str, Any]:
    """Evaluate on Wonderland dev subset."""
    import torch
    from collections import Counter

    # Load dev records
    dev_records = []
    dev_path = STAGE3_DATA_DIR / "dev.jsonl"
    with open(dev_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dev_records.append(json.loads(line))

    model, tokenizer, stop_ids = _load_model_for_eval(config, adapter_path)

    per_sample: list[dict] = []
    by_task: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})

    for rec in dev_records:
        task_type = rec.get("task_type", "unknown")
        sample_type = rec.get("sample_type", "unknown")
        # Get user message only
        user_msgs = [m for m in rec["messages"] if m["role"] == "user"]
        expected = rec["messages"][-1]["content"].strip() if rec["messages"] else ""

        # Determine thinking mode
        wants_think = sample_type in ("wonderland_compressed_cot", "stage2_thinking_replay")
        generated = _generate(
            model, tokenizer, stop_ids,
            user_msgs,
            max_new_tokens=256,
            enable_thinking=wants_think,
        )

        # Extract final answer (after </think> if present)
        if "</think>" in generated:
            final_answer = generated.split("</think>")[-1].strip()
            think_content = generated.split("<think>", 1)[-1].split("</think>", 1)[0].strip()
        else:
            final_answer = generated.strip()
            think_content = ""

        # Simple answer comparison (normalize whitespace)
        expected_normalized = " ".join(expected.split())
        generated_normalized = " ".join(generated.split())
        final_normalized = " ".join(final_answer.split())
        is_exact = final_normalized == expected_normalized

        has_think = "<think>" in generated
        has_close = "</think>" in generated

        by_task[task_type]["total"] += 1
        if is_exact:
            by_task[task_type]["correct"] += 1

        per_sample.append({
            "id": rec.get("id", ""),
            "task_type": task_type,
            "sample_type": sample_type,
            "expected": expected[:200],
            "generated": generated[:300],
            "final_answer": final_answer[:200],
            "is_exact": is_exact,
            "has_think": has_think,
            "has_close_think": has_close,
            "think_len": len(think_content),
            "wants_think": wants_think,
        })

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Aggregate stats
    total = len(dev_records)
    exact = sum(1 for r in per_sample if r["is_exact"])
    spurious_think = sum(1 for r in per_sample if r["has_think"] and not r["wants_think"])
    missing_think = sum(1 for r in per_sample if not r["has_think"] and r["wants_think"])

    # Per task_type stats
    task_stats = {}
    for tt, counts in sorted(by_task.items()):
        task_stats[tt] = {
            "correct": counts["correct"],
            "total": counts["total"],
            "accuracy": counts["correct"] / counts["total"] if counts["total"] else 0,
        }

    return {
        "total": total,
        "exact_match": exact,
        "exact_match_rate": round(exact / total, 4) if total else 0,
        "spurious_think": spurious_think,
        "missing_think": missing_think,
        "task_type_breakdown": task_stats,
        "samples": per_sample[:10],  # First 10 for report
    }


def phase5c_strict_regression(config: dict[str, Any], adapter_path: str) -> dict[str, Any]:
    """Stage 1.5 strict regression: check no thinking on strict prompts."""
    import torch

    # Load stage1_5 strict dev
    strict_path = Path("data/instruction/stage1_5/validation.jsonl")
    if not strict_path.is_file():
        print("    WARNING: stage1_5 validation.jsonl not found, skipping")
        return {"status": "skipped", "reason": "stage1_5 validation.jsonl not found"}

    records = []
    with open(strict_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # Use first 40 records
    records = records[:40]

    model, tokenizer, stop_ids = _load_model_for_eval(config, adapter_path)

    exact = 0
    spurious_think = 0
    stop_ok = 0
    per_sample: list[dict] = []

    for rec in records:
        messages = rec.get("messages", [])
        user_msgs = [m for m in messages if m["role"] != "assistant"]
        expected = messages[-1]["content"].strip() if messages else ""

        generated = _generate(model, tokenizer, stop_ids, user_msgs, max_new_tokens=128, enable_thinking=False)

        gen_clean = generated.strip()
        has_think = "<think>" in generated
        if has_think:
            spurious_think += 1

        if gen_clean == expected:
            exact += 1

        tok_count = len(tokenizer.encode(generated))
        if tok_count < 128:
            stop_ok += 1

        per_sample.append({
            "id": rec.get("id", ""),
            "expected": expected[:150],
            "generated": generated[:150],
            "exact_match": gen_clean == expected,
            "spurious_think": has_think,
            "stop_ok": tok_count < 128,
        })

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    total = len(records)
    return {
        "total": total,
        "exact_match": exact,
        "exact_match_rate": round(exact / total, 4) if total else 0,
        "spurious_think": spurious_think,
        "spurious_think_rate": round(spurious_think / total, 4) if total else 0,
        "stop_success": stop_ok,
        "stop_success_rate": round(stop_ok / total, 4) if total else 0,
        "samples": per_sample[:8],
    }


def phase5d_protocol_regression(config: dict[str, Any], adapter_path: str) -> dict[str, Any]:
    """Stage 2 protocol regression: check think tags when requested."""
    import torch

    stage2_path = Path("data/instruction/stage2_thinking/validation.jsonl")
    if not stage2_path.is_file():
        print("    WARNING: stage2 validation.jsonl not found, skipping")
        return {"status": "skipped", "reason": "stage2 validation.jsonl not found"}

    records = []
    with open(stage2_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # Filter by metadata.thinking flag (True=thinking requested, False=no thinking)
    think_records = [r for r in records if r.get("metadata", {}).get("thinking", False)][:30]
    no_think_records = [r for r in records if not r.get("metadata", {}).get("thinking", False)][:30]

    model, tokenizer, stop_ids = _load_model_for_eval(config, adapter_path)

    results_think: list[dict] = []
    results_nothink: list[dict] = []

    # Evaluate thinking requests
    for rec in think_records:
        messages = rec.get("messages", [])
        user_msgs = [m for m in messages if m["role"] != "assistant"]
        wants_think = rec.get("metadata", {}).get("thinking", True)
        generated = _generate(model, tokenizer, stop_ids, user_msgs, max_new_tokens=256, enable_thinking=True)
        has_think = "<think>" in generated
        has_close = "</think>" in generated
        results_think.append({
            "id": rec.get("id", ""),
            "generated": generated[:200],
            "has_think": has_think,
            "has_close": has_close,
            "both_tags": has_think and has_close,
        })

    # Evaluate no-thinking requests
    for rec in no_think_records:
        messages = rec.get("messages", [])
        user_msgs = [m for m in messages if m["role"] != "assistant"]
        generated = _generate(model, tokenizer, stop_ids, user_msgs, max_new_tokens=256, enable_thinking=False)
        has_think = "<think>" in generated
        results_nothink.append({
            "id": rec.get("id", ""),
            "generated": generated[:200],
            "has_think": has_think,
            "no_spurious": not has_think,
        })

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    think_total = len(think_records)
    think_ok = sum(1 for r in results_think if r["both_tags"])
    no_think_total = len(results_nothink)
    no_think_ok = sum(1 for r in results_nothink if r["no_spurious"])

    return {
        "thinking": {
            "total": think_total,
            "both_tags": think_ok,
            "rate": round(think_ok / think_total, 4) if think_total else 0,
        },
        "no_thinking": {
            "total": no_think_total,
            "no_spurious_think": no_think_ok,
            "rate": round(no_think_ok / no_think_total, 4) if no_think_total else 0,
        },
        "think_samples": results_think[:5],
        "no_think_samples": results_nothink[:5],
    }


OPEN_PROMPTS = [
    "Explain the difference between LoRA and full fine-tuning in 2-3 sentences.",
    "Summarize: The rapid advancement of large language models has transformed NLP. These models, trained on vast corpora of text, can perform tasks from translation to code generation. However, their deployment raises concerns about computational cost, data privacy, and potential misuse.",
    "Give three reasons why retrieval-augmented generation helps large language models.",
    "Write a short Python function that reverses a list without using built-in reverse methods.",
    "What is the capital of France? Explain why it became the capital.",
    "Describe the water cycle in 3-4 sentences.",
    "Compare cats and dogs as pets. List 2 pros and 2 cons for each.",
    "Explain what a neural network is to a 10-year-old.",
    "What are the three main branches of the U.S. government and what does each do?",
    "What is the difference between SQL and NoSQL databases? Give one use case for each.",
    "Explain the concept of recursion with a simple example.",
    "What is the purpose of version control in software development?",
    "Name three renewable energy sources and briefly describe how each works.",
    "What does 'open source' mean in the context of software? Give an example.",
    "Explain why the sky is blue in simple terms.",
    "What are microservices? Give one advantage and one disadvantage.",
    "Describe a simple algorithm for sorting a list of numbers.",
    "Rewrite professionally: 'Hey can u pls send me the report by tomorrow thx.'",
    "Write a short haiku about programming.",
    "If you could only recommend one book to someone learning programming, what would it be and why?",
]


def phase5e_open_regression(config: dict[str, Any], adapter_path: str) -> dict[str, Any]:
    """Open replay regression: check general answering quality."""
    import torch

    model, tokenizer, stop_ids = _load_model_for_eval(config, adapter_path)

    prompts = OPEN_PROMPTS[:20]
    per_sample: list[dict] = []

    too_short_count = 0
    stop_ok = 0
    spurious_think = 0

    for i, prompt_text in enumerate(prompts):
        history = [{"role": "user", "content": prompt_text}]
        prompt_str = tokenizer.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=[tokenizer.eos_token_id, stop_ids.get("im_end"), stop_ids.get("endoftext")],
            )

        response = tokenizer.decode(
            output[0][prompt_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False
        ).strip()

        word_count = len(response.split())
        has_think = "<think>" in response
        tok_count = len(tokenizer.encode(response))

        if word_count <= 3:
            too_short_count += 1
        if tok_count < 256:
            stop_ok += 1
        if has_think:
            spurious_think += 1

        per_sample.append({
            "index": i,
            "prompt": prompt_text[:100],
            "response": response[:300],
            "word_count": word_count,
            "has_think": has_think,
            "too_short": word_count <= 3,
            "stop_ok": tok_count < 256,
        })

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    total = len(prompts)
    word_counts = [r["word_count"] for r in per_sample]

    return {
        "total": total,
        "too_short": too_short_count,
        "too_short_rate": round(too_short_count / total, 4) if total else 0,
        "spurious_think": spurious_think,
        "spurious_think_rate": round(spurious_think / total, 4) if total else 0,
        "stop_success": stop_ok,
        "stop_success_rate": round(stop_ok / total, 4) if total else 0,
        "word_count_mean": round(sum(word_counts) / len(word_counts), 1) if word_counts else 0,
        "word_count_min": min(word_counts) if word_counts else 0,
        "word_count_max": max(word_counts) if word_counts else 0,
        "samples": per_sample,
    }


# ── Phase 6: Reports ─────────────────────────────────────────────────────────


def phase6_reports(
    manifest: dict[str, Any],
    audit: dict[str, Any],
    train_metrics: dict[str, Any],
    eval_results: dict[str, Any],
) -> None:
    """Write smoke_train_report.json and smoke_eval_report.md."""
    print("[phase6] Writing reports...")

    # ── smoke_train_report.json ──
    # Extract key training metrics
    train_dict = train_metrics.get("train", {})
    val_dict = train_metrics.get("validation", {})

    train_report = {
        "stage": "stage3_wonderland_coldstart",
        "mode": "smoke",
        "data": {
            "source_dir": str(STAGE3_DATA_DIR),
            "smoke_dir": str(SMOKE_DATA_DIR),
            "manifest": manifest,
        },
        "sampling": {
            "total": 300,
            "breakdown": SAMPLE_TYPE_LIMITS,
        },
        "label_audit": {
            "train_eligible": audit["token_report"]["splits"]["train"]["eligible_count"],
            "train_total": audit["token_report"]["splits"]["train"]["count"],
            "train_excluded": audit["token_report"]["splits"]["train"]["excluded_count"],
            "validation_eligible": audit["token_report"]["splits"]["validation"]["eligible_count"],
            "batch_audit_passed": audit["batch_audit"]["status"] == "passed",
        },
        "training": {
            "init_adapter": str(STAGE2_ADAPTER),
            "adapter_output": str(SMOKE_OUTPUT_DIR / "smoke" / "adapter"),
            "global_step": train_metrics.get("global_step"),
            "train_loss": train_dict.get("train_loss"),
            "train_runtime": train_dict.get("train_runtime"),
            "eval_loss": val_dict.get("eval_loss"),
            "eval_runtime": val_dict.get("eval_runtime"),
            "peak_gpu_memory_bytes": train_metrics.get("peak_gpu_memory_bytes"),
            "supervised_tokens_seen": train_metrics.get("supervised_tokens_seen"),
            "total_tokens_seen": train_metrics.get("total_tokens_seen"),
            "elapsed_seconds": train_metrics.get("elapsed_seconds"),
            "samples_per_second": train_dict.get("train_samples_per_second"),
            "tokens_per_second": int(
                train_metrics.get("supervised_tokens_seen", 0) / train_metrics.get("elapsed_seconds", 1)
            ) if train_metrics.get("elapsed_seconds") else 0,
        },
    }

    report_path = SMOKE_OUTPUT_DIR / "smoke_train_report.json"
    write_json(report_path, train_report)
    print(f"  Written: {report_path}")

    # ── smoke_eval_report.md ──
    lines = [
        "# Stage 3 Wonderland Cold Start - Smoke Eval Report",
        "",
        f"**Training adapter:** `{SMOKE_OUTPUT_DIR / 'smoke' / 'adapter'}`",
        f"**Init adapter:** `{STAGE2_ADAPTER}`",
        "",
        "---",
        "",
        "## 1. Adapter Reload Sanity",
        "",
    ]

    sanity = eval_results["adapter_sanity"]
    lines.append(f"- **Passed:** {sanity['passed']}")
    lines.append(f"- **Test prompt:** {sanity['test_prompt']}")
    lines.append(f"- **Generated:** `{sanity['generated']}`")
    lines.append(f"- **Length:** {sanity['generated_length']} chars")
    lines.append(f"- **Time:** {sanity['elapsed_sec']}s")
    lines.append("")

    # Wonderland dev
    wdev = eval_results["wonderland_dev"]
    lines.append("---")
    lines.append("")
    lines.append("## 2. Wonderland Dev Subset")
    lines.append("")
    lines.append(f"- **Total:** {wdev['total']}")
    lines.append(f"- **Exact match:** {wdev['exact_match']} ({wdev['exact_match_rate']:.2%})")
    lines.append(f"- **Spurious think:** {wdev['spurious_think']}")
    lines.append(f"- **Missing think:** {wdev['missing_think']}")
    lines.append("")
    lines.append("### Task-Type Breakdown")
    lines.append("")
    lines.append("| Task Type | Correct | Total | Accuracy |")
    lines.append("|-----------|---------|-------|----------|")
    for tt, stats in sorted(wdev.get("task_type_breakdown", {}).items()):
        lines.append(f"| {tt} | {stats['correct']} | {stats['total']} | {stats['accuracy']:.2%} |")
    lines.append("")

    # Sample outputs
    lines.append("### Sample Predictions")
    lines.append("")
    for i, s in enumerate(wdev.get("samples", [])[:5]):
        lines.append(f"#### {i + 1}. {s.get('task_type', '')} ({s.get('sample_type', '')})")
        lines.append(f"- **Expected:** `{s['expected'][:120]}`")
        lines.append(f"- **Generated:** `{s['generated'][:200]}`")
        lines.append(f"- **Exact:** {s['is_exact']} | **Think:** {s['has_think']}")
        lines.append("")

    # Stage 1.5 strict regression
    s15 = eval_results["stage1_5_strict"]
    lines.append("---")
    lines.append("")
    lines.append("## 3. Stage 1.5 Strict Regression")
    lines.append("")
    if s15.get("status") == "skipped":
        lines.append(f"**Skipped:** {s15.get('reason', '')}")
    else:
        lines.append(f"- **Total:** {s15['total']}")
        lines.append(f"- **Exact match:** {s15['exact_match']} ({s15['exact_match_rate']:.2%})")
        lines.append(f"- **Spurious think:** {s15['spurious_think']} ({s15['spurious_think_rate']:.2%})")
        lines.append(f"- **Stop success:** {s15['stop_success']} ({s15['stop_success_rate']:.2%})")
    lines.append("")

    # Stage 2 protocol regression
    s2 = eval_results["stage2_protocol"]
    lines.append("---")
    lines.append("")
    lines.append("## 4. Stage 2 Protocol Regression")
    lines.append("")
    if s2.get("status") == "skipped":
        lines.append(f"**Skipped:** {s2.get('reason', '')}")
    else:
        think = s2["thinking"]
        nothink = s2["no_thinking"]
        lines.append(f"- **Thinking requests:** {think['both_tags']}/{think['total']} ({think['rate']:.2%}) with both tags")
        lines.append(f"- **No-think requests:** {nothink['no_spurious_think']}/{nothink['total']} ({nothink['rate']:.2%}) without spurious think")
    lines.append("")

    # Open regression
    open_reg = eval_results["open_regression"]
    lines.append("---")
    lines.append("")
    lines.append("## 5. Open Replay Regression")
    lines.append("")
    lines.append(f"- **Total:** {open_reg['total']}")
    lines.append(f"- **Too short (<4 words):** {open_reg['too_short']} ({open_reg['too_short_rate']:.2%})")
    lines.append(f"- **Spurious think:** {open_reg['spurious_think']} ({open_reg['spurious_think_rate']:.2%})")
    lines.append(f"- **Stop success:** {open_reg['stop_success']} ({open_reg['stop_success_rate']:.2%})")
    lines.append(f"- **Word count:** mean={open_reg['word_count_mean']}, min={open_reg['word_count_min']}, max={open_reg['word_count_max']}")
    lines.append("")
    lines.append("### Sample Outputs")
    lines.append("")
    for s in open_reg.get("samples", [])[:5]:
        lines.append(f"**Q:** {s['prompt']}")
        lines.append(f"**A:** `{s['response'][:200]}`")
        lines.append(f"*Words: {s['word_count']} | Think: {s['has_think']} | TooShort: {s['too_short']}*")
        lines.append("")

    # Write report
    md_path = SMOKE_OUTPUT_DIR / "smoke_eval_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Written: {md_path}")


# ── Main ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3 smoke training + eval")
    parser.add_argument("--config", type=Path, default=Path("configs/stage3_wonderland_coldstart.yaml"))
    parser.add_argument("--phase", choices=["all", "freeze", "audit", "sample", "train", "eval", "report"],
                        default="all", help="Run specific phase(s)")
    parser.add_argument("--skip-train", action="store_true", help="Skip training (eval only)")
    parser.add_argument("--set", action="append", default=[], dest="overrides",
                        help="Override YAML value, e.g. training.learning_rate=5e-5")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    if args.overrides:
        config = apply_overrides(config, args.overrides)

    print("=" * 60)
    print("Stage 3 Wonderland Cold Start - Smoke Training")
    print(f"Config: {args.config}")
    print(f"Phase: {args.phase}")
    print("=" * 60)

    # ── Phase 1: Freeze data ──
    manifest = {}
    if args.phase in ("all", "freeze"):
        manifest = phase1_freeze()

    # ── Phase 2: Sample smoke data ──
    # Sample first, then audit the smoke data
    if args.phase in ("all", "sample"):
        sampled = phase3_sample()
        phase3_write_smoke_data(sampled)

    # ── Phase 3: Label audit smoke data ──
    audit = {}
    if args.phase in ("all", "audit"):
        # Update config to point to smoke data for audit
        smoke_config = dict(config)
        smoke_config["data"] = dict(config["data"])
        smoke_config["data"]["output_dir"] = str(SMOKE_DATA_DIR)
        audit = phase2_label_audit(smoke_config)

    # ── Phase 4: Training ──
    train_metrics = {}
    if args.phase in ("all", "train"):
        if args.skip_train:
            print("[phase4] Skipping training (--skip-train)")
        else:
            # Use smoke config for training
            smoke_config = dict(config)
            smoke_config["data"] = dict(config["data"])
            smoke_config["data"]["output_dir"] = str(SMOKE_DATA_DIR)
            train_metrics = phase4_train(smoke_config)

    # ── Phase 5: Evaluation ──
    eval_results = {}
    if args.phase in ("all", "eval"):
        eval_results = phase5_eval(config)

    # ── Phase 6: Reports ──
    if args.phase in ("all", "report"):
        if not manifest:
            manifest = phase1_freeze()
        if not audit:
            audit_data = json.loads((SMOKE_DATA_DIR / "token_report.json").read_text())
            batch_data = json.loads((SMOKE_DATA_DIR / "batch_audit.json").read_text())
            audit = {"token_report": audit_data, "batch_audit": batch_data}
        if not train_metrics:
            train_json = SMOKE_OUTPUT_DIR / "smoke" / "train_metrics.json"
            if train_json.is_file():
                train_metrics = json.loads(train_json.read_text(encoding="utf-8"))
            else:
                train_metrics = {}
        if not eval_results:
            eval_results = phase5_eval(config)

        phase6_reports(manifest, audit, train_metrics, eval_results)

    print()
    print("Done. Outputs:")
    print(f"  Data: {SMOKE_DATA_DIR}/")
    print(f"  Training: {SMOKE_OUTPUT_DIR}/")
    print(f"  Reports: {SMOKE_OUTPUT_DIR}/smoke_train_report.json")
    print(f"           {SMOKE_OUTPUT_DIR}/smoke_eval_report.md")


if __name__ == "__main__":
    main()
