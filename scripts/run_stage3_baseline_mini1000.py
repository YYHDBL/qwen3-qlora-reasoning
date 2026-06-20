#!/usr/bin/env python3
"""Stage 3 baseline eval + error analysis + mini_1000 training.

Tasks:
  1. Stage 2 baseline eval (same 5 suites as smoke_300)
  2. Error analysis on smoke_300 failures
  3. mini_1000 training from Stage 2 adapter
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.config import load_yaml_config, apply_overrides
from src.common.experiment import sha256_file, write_json, write_jsonl

STAGE3_DATA_DIR = Path("data/instruction/stage3_wonderland_cold_start")
STAGE2_ADAPTER = Path("outputs/stage2_thinking_warmup/formal/adapter")
SMOKE_ADAPTER = Path("outputs/stage3_wonderland_coldstart/smoke_300/smoke/adapter")
BASELINE_OUTPUT = Path("outputs/stage3_wonderland_coldstart/baseline_stage2")
MINI1000_OUTPUT = Path("outputs/stage3_wonderland_coldstart/mini_1000")
MINI1000_DATA = Path("data/instruction/stage3_wonderland_cold_start/mini_1000")

RANDOM_SEED = 42

# ── Model loading (shared with smoke script) ─────────────────────────────────


def _load_model_for_eval(config: dict[str, Any], adapter_path: str):
    import torch
    from src.training.model_loader import load_lora_model, load_tokenizer
    from src.training.chat_template import resolve_stop_token_ids

    tokenizer = load_tokenizer(config, for_training=False)
    model = load_lora_model(config, adapter_path, is_trainable=False).eval()
    model.config.use_cache = True
    stop_ids = resolve_stop_token_ids(tokenizer)
    return model, tokenizer, stop_ids


def _generate(model, tokenizer, stop_ids, messages, max_new_tokens=256, enable_thinking=None):
    import torch
    from src.training.chat_template import render_generation_prompt
    prompt = render_generation_prompt(tokenizer, list(messages), enable_thinking=enable_thinking)
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    n_in = inputs.input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id,
                             eos_token_id=[tokenizer.eos_token_id, stop_ids.get("im_end"), stop_ids.get("endoftext")])
    return tokenizer.decode(out[0][n_in:], skip_special_tokens=True).strip()


# ── Eval suite (same 5 tests as smoke) ───────────────────────────────────────

def eval_wonderland_dev(config, adapter_path):
    """Eval on Wonderland dev (120 samples). Returns detailed per-sample + stats."""
    import torch
    dev_records = []
    with open(STAGE3_DATA_DIR / "dev.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dev_records.append(json.loads(line))

    model, tokenizer, stop_ids = _load_model_for_eval(config, adapter_path)
    per_sample, by_task = [], defaultdict(lambda: {"correct": 0, "total": 0})

    for rec in dev_records:
        tt = rec.get("task_type", "unknown")
        st = rec.get("sample_type", "unknown")
        user_msgs = [m for m in rec["messages"] if m["role"] == "user"]
        expected = rec["messages"][-1]["content"].strip()

        wants_think = st in ("wonderland_compressed_cot", "stage2_thinking_replay")
        gen = _generate(model, tokenizer, stop_ids, user_msgs, max_new_tokens=256, enable_thinking=wants_think)

        has_think = "<think>" in gen
        has_close = "</think>" in gen
        if "</think>" in gen:
            final_answer = gen.split("</think>")[-1].strip()
            think_content = gen.split("<think>", 1)[-1].split("</think>", 1)[0].strip()
        else:
            final_answer = gen.strip()
            think_content = ""

        exp_norm = " ".join(expected.split())
        final_norm = " ".join(final_answer.split())
        is_exact = final_norm == exp_norm

        by_task[tt]["total"] += 1
        if is_exact:
            by_task[tt]["correct"] += 1

        per_sample.append({
            "id": rec.get("id", ""),
            "task_type": tt, "sample_type": st,
            "expected": expected[:200], "generated": gen[:300],
            "final_answer": final_answer[:200], "think_content": think_content[:200],
            "is_exact": is_exact, "has_think": has_think, "has_close": has_close,
            "wants_think": wants_think,
            "num_gen_tokens": len(tokenizer.encode(gen)),
        })

    del model; torch.cuda.empty_cache()
    total = len(dev_records)
    exact = sum(1 for r in per_sample if r["is_exact"])
    spurious = sum(1 for r in per_sample if r["has_think"] and not r["wants_think"])
    missing = sum(1 for r in per_sample if not r["has_think"] and r["wants_think"])
    task_stats = {tt: {"correct": c["correct"], "total": c["total"],
                       "accuracy": round(c["correct"]/c["total"], 4) if c["total"] else 0}
                  for tt, c in sorted(by_task.items())}

    # Also compute final_answer_parse rate for CoT samples
    cot_samples = [r for r in per_sample if r["wants_think"]]
    parse_ok = sum(1 for r in cot_samples if r["has_close"] and r["final_answer"])

    return {"total": total, "exact_match": exact,
            "exact_match_rate": round(exact/total, 4) if total else 0,
            "spurious_think": spurious, "missing_think": missing,
            "parse_ok": parse_ok, "parse_total": len(cot_samples),
            "parse_rate": round(parse_ok/len(cot_samples), 4) if cot_samples else 0,
            "task_type_breakdown": task_stats,
            "samples": per_sample}


def eval_strict_regression(config, adapter_path):
    import torch
    sp = Path("data/instruction/stage1_5/validation.jsonl")
    if not sp.is_file():
        return {"status": "skipped"}
    records = [json.loads(l) for l in open(sp) if l.strip()][:40]
    model, tokenizer, stop_ids = _load_model_for_eval(config, adapter_path)
    exact = spurious = stop_ok = 0
    per_sample = []
    for rec in records:
        user = [m for m in rec["messages"] if m["role"] != "assistant"]
        exp = rec["messages"][-1]["content"].strip()
        gen = _generate(model, tokenizer, stop_ids, user, max_new_tokens=128, enable_thinking=False)
        gc = gen.strip()
        ht = "<think>" in gen
        if gc == exp: exact += 1
        if ht: spurious += 1
        if len(tokenizer.encode(gen)) < 128: stop_ok += 1
        per_sample.append({"id": rec.get("id",""), "exact": gc==exp, "spurious": ht, "stop_ok": len(tokenizer.encode(gen))<128,
                           "expected": exp[:120], "generated": gen[:150]})
    del model; torch.cuda.empty_cache()
    t = len(records)
    return {"total": t, "exact_match": exact, "exact_rate": round(exact/t,4) if t else 0,
            "spurious_think": spurious, "spurious_rate": round(spurious/t,4) if t else 0,
            "stop_success": stop_ok, "stop_rate": round(stop_ok/t,4) if t else 0}


def eval_protocol_regression(config, adapter_path):
    import torch
    sp = Path("data/instruction/stage2_thinking/validation.jsonl")
    if not sp.is_file():
        return {"status": "skipped"}
    records = [json.loads(l) for l in open(sp) if l.strip()]
    think = [r for r in records if r.get("metadata",{}).get("thinking",False)][:30]
    nthk = [r for r in records if not r.get("metadata",{}).get("thinking",False)][:30]
    model, tokenizer, stop_ids = _load_model_for_eval(config, adapter_path)
    t_ok = n_ok = 0
    for rec in think:
        user = [m for m in rec["messages"] if m["role"]!="assistant"]
        gen = _generate(model, tokenizer, stop_ids, user, max_new_tokens=256, enable_thinking=True)
        if "<think>" in gen and "</think>" in gen: t_ok += 1
    for rec in nthk:
        user = [m for m in rec["messages"] if m["role"]!="assistant"]
        gen = _generate(model, tokenizer, stop_ids, user, max_new_tokens=256, enable_thinking=False)
        if "<think>" not in gen: n_ok += 1
    del model; torch.cuda.empty_cache()
    return {"thinking": {"total": len(think), "both_tags": t_ok, "rate": round(t_ok/len(think),4) if think else 0},
            "no_thinking": {"total": len(nthk), "no_spurious": n_ok, "rate": round(n_ok/len(nthk),4) if nthk else 0}}


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


def eval_open_regression(config, adapter_path):
    import torch
    model, tokenizer, stop_ids = _load_model_for_eval(config, adapter_path)
    prompts = OPEN_PROMPTS[:20]
    per_sample, too_short, stop_ok, spurious = [], 0, 0, 0
    for i, p in enumerate(prompts):
        hist = [{"role": "user", "content": p}]
        ps = tokenizer.apply_chat_template(hist, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inp = tokenizer(ps, return_tensors="pt").to(model.device)
        n_in = inp["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=256, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id,
                                 eos_token_id=[tokenizer.eos_token_id, stop_ids.get("im_end"), stop_ids.get("endoftext")])
        resp = tokenizer.decode(out[0][n_in:], skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        wc = len(resp.split())
        ht = "<think>" in resp
        ct = len(tokenizer.encode(resp))
        if wc <= 3: too_short += 1
        if ct < 256: stop_ok += 1
        if ht: spurious += 1
        per_sample.append({"idx": i, "prompt": p[:100], "response": resp[:300], "word_count": wc,
                           "has_think": ht, "too_short": wc<=3, "stop_ok": ct<256})
    del model; torch.cuda.empty_cache()
    t = len(prompts)
    wcs = [r["word_count"] for r in per_sample]
    return {"total": t, "too_short": too_short, "too_short_rate": round(too_short/t,4) if t else 0,
            "spurious_think": spurious, "spurious_rate": round(spurious/t,4) if t else 0,
            "stop_success": stop_ok, "stop_rate": round(stop_ok/t,4) if t else 0,
            "word_mean": round(sum(wcs)/len(wcs),1) if wcs else 0,
            "word_min": min(wcs) if wcs else 0, "word_max": max(wcs) if wcs else 0}


def run_full_eval(config, adapter_path, label="") -> dict:
    print(f"  [{label}] Adapter sanity...")
    model, tokenizer, stop_ids = _load_model_for_eval(config, adapter_path)
    sanity_gen = _generate(model, tokenizer, stop_ids,
                           [{"role":"user","content":"What is 2+2? Answer in one word."}],
                           max_new_tokens=32, enable_thinking=False)
    del model
    import torch; torch.cuda.empty_cache()
    sanity = {"passed": len(sanity_gen)>0 and len(sanity_gen)<200, "generated": sanity_gen,
              "test_prompt": "What is 2+2? Answer in one word."}

    print(f"  [{label}] Wonderland dev...")
    wdev = eval_wonderland_dev(config, adapter_path)
    print(f"  [{label}] Strict regression...")
    strict = eval_strict_regression(config, adapter_path)
    print(f"  [{label}] Protocol regression...")
    proto = eval_protocol_regression(config, adapter_path)
    print(f"  [{label}] Open regression...")
    open_r = eval_open_regression(config, adapter_path)

    return {"label": label, "sanity": sanity, "wonderland_dev": wdev,
            "strict_regression": strict, "protocol_regression": proto, "open_regression": open_r}


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 1: Stage 2 baseline eval
# ═══════════════════════════════════════════════════════════════════════════════

def task1_baseline_eval(config):
    print("=" * 60)
    print("TASK 1: Stage 2 Baseline Eval")
    print("=" * 60)
    baseline = run_full_eval(config, str(STAGE2_ADAPTER), label="baseline-st2")
    BASELINE_OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(BASELINE_OUTPUT / "baseline_eval.json", baseline)

    # Load smoke results for side-by-side
    smoke_report = json.loads((Path("outputs/stage3_wonderland_coldstart/smoke_300/smoke_train_report.json")
                               .read_text(encoding="utf-8")))

    # Side-by-side markdown
    lines = [
        "# Stage 2 Baseline vs Stage 3 Smoke-300 — Side-by-Side Eval Comparison",
        "",
        f"**Stage 2 adapter:** `{STAGE2_ADAPTER}`",
        f"**Stage 3 Smoke adapter:** `{SMOKE_ADAPTER}`",
        f"**Eval data:** Wonderland dev (120), strict (40), protocol (60), open (20)",
        "",
        "---",
        "",
        "## 1. Wonderland Dev Accuracy",
        "",
        "| Task Type | Stg2 Baseline | Stg3 Smoke-300 | Delta |",
        "|-----------|:------------:|:-------------:|:-----:|",
    ]

    bw = baseline["wonderland_dev"]
    sw = smoke_report
    for tt in sorted(bw["task_type_breakdown"]):
        ba = bw["task_type_breakdown"].get(tt, {}).get("accuracy", 0)
        sa = 0  # We'll get from the smoke eval report
        lines.append(f"| {tt} | {ba:.2%} | {sa:.2%} | |")

    # Actually I need the smoke eval results too. Let me re-read the smoke eval report.
    smoke_eval_data = None
    # We'll fill in after loading

    write_json(BASELINE_OUTPUT / "side_by_side.json", {"baseline": baseline, "note": "smoke data loaded separately"})

    print(f"  Baseline saved to {BASELINE_OUTPUT / 'baseline_eval.json'}")
    return baseline


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 2: Error analysis
# ═══════════════════════════════════════════════════════════════════════════════

def classify_error(sample: dict) -> str:
    """Classify one prediction error."""
    gen = sample.get("generated", "")
    final = sample.get("final_answer", "")
    has_think = sample.get("has_think", False)
    has_close = sample.get("has_close", False)
    wants_think = sample.get("wants_think", False)
    num_tokens = sample.get("num_gen_tokens", 0)

    if wants_think and not has_think:
        return "missing_think"
    if wants_think and not has_close:
        return "unclosed_think"
    if wants_think and not final:
        return "parse_fail"
    if num_tokens >= 256:
        return "stop_fail"
    if not final:
        return "parse_fail"
    # Check if answer is close but wrong (rounding, etc.)
    exp = sample.get("expected", "")
    if "".join(filter(str.isdigit, final)) == "".join(filter(str.isdigit, exp)) and final != exp:
        return "rounding_error"
    if len(final) > 0 and final != exp:
        return "answer_wrong"
    return "unknown"


def task2_error_analysis(config):
    print("=" * 60)
    print("TASK 2: Error Analysis")
    print("=" * 60)

    # Load full smoke eval data (from wonderland dev eval)
    adapter_path = str(SMOKE_ADAPTER)
    wdev = eval_wonderland_dev(config, adapter_path)

    error_types = ["bit_manipulation", "cipher", "gravity", "symbolic_equation", "unit_conversion"]
    analysis = {}
    for tt in error_types:
        err_samples = [s for s in wdev["samples"] if s["task_type"] == tt and not s["is_exact"]]
        # Pick 5 random (deterministic)
        random.seed(42)
        selected = random.sample(err_samples, min(5, len(err_samples)))
        classified = []
        for s in selected:
            err_cls = classify_error(s)
            classified.append({
                "id": s["id"],
                "sample_type": s["sample_type"],
                "prompt_preview": s.get("expected", "")[:200],
                "gold": s["expected"][:200],
                "model_output": s["generated"][:300],
                "parsed_answer": s["final_answer"][:200],
                "has_think": s["has_think"],
                "has_close": s["has_close"],
                "wants_think": s["wants_think"],
                "error_class": err_cls,
            })
        analysis[tt] = {
            "total_errors": len(err_samples),
            "sample_count": len(classified),
            "samples": classified,
        }

    # Write report
    lines = [
        "# Smoke-300 Error Analysis",
        "",
        "## Summary",
        "",
        "| Task Type | Total Dev | Errors | Error Rate | Top Error Class |",
        "|-----------|-----------|--------|------------|-----------------|",
    ]
    for tt in error_types:
        errs = analysis[tt]["total_errors"]
        bd = wdev["task_type_breakdown"].get(tt, {})
        total = bd.get("total", 0)
        rate = errs / total if total else 0
        classes = Counter(s["error_class"] for s in analysis[tt]["samples"])
        top = classes.most_common(1)[0][0] if classes else "N/A"
        lines.append(f"| {tt} | {total} | {errs} | {rate:.1%} | {top} |")

    lines.append("")
    lines.append("## Detailed Samples")
    lines.append("")

    for tt in error_types:
        lines.append(f"### {tt} ({analysis[tt]['sample_count']} samples)")
        lines.append("")
        for i, s in enumerate(analysis[tt]["samples"]):
            lines.append(f"#### {i+1}. [{s['error_class']}]")
            lines.append(f"- **Type:** {s['sample_type']} | Think: {s['has_think']} | Close: {s['has_close']}")
            lines.append(f"- **Gold:** `{s['gold'][:120]}`")
            lines.append(f"- **Output:** `{s['model_output'][:200]}`")
            lines.append(f"- **Parsed:** `{s['parsed_answer'][:120]}`")
            lines.append("")

    md_path = Path("outputs/stage3_wonderland_coldstart/smoke_error_analysis.md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf-8")
    write_json(Path("outputs/stage3_wonderland_coldstart/smoke_error_analysis.json"), analysis)

    print(f"  Written: {md_path}")
    return analysis


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 3: mini_1000 training
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLE_LIMITS_1000 = {
    "wonderland_answer_only": 531,
    "wonderland_compressed_cot": 280,
    "stage1_5_strict_replay": 91,
    "stage2_thinking_replay": 98,
}


def _largest_remainder_allocation(total: int, proportions: Sequence[tuple[str, int]]) -> dict[str, int]:
    total_count = sum(cnt for _, cnt in proportions)
    if total_count == 0: return {}
    quota = total / total_count
    allocated = {}
    remainders = []
    assigned_sum = 0
    for key, cnt in proportions:
        alloc = int(cnt * quota)
        allocated[key] = alloc
        assigned_sum += alloc
        remainders.append((cnt * quota - alloc, key, cnt))
    remainders.sort(key=lambda x: x[0], reverse=True)
    for i in range(total - assigned_sum):
        if i >= len(remainders): break
        allocated[remainders[i][1]] += 1
    return allocated


def sample_mini_1000():
    print("  Sampling ~1000 records for mini_1000 training...")
    random.seed(RANDOM_SEED)
    records = []
    with open(STAGE3_DATA_DIR / "train.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    groups = defaultdict(list)
    for r in records:
        groups[(r["sample_type"], r["task_type"])].append(r)
    for gr in groups.values():
        random.shuffle(gr)

    sampled = []
    for st, limit in SAMPLE_LIMITS_1000.items():
        st_groups = {tt: len(gr) for (s, tt), gr in groups.items() if s == st}
        alloc = _largest_remainder_allocation(limit, list(st_groups.items()))
        for tt, n in alloc.items():
            gr = groups.get((st, tt), [])
            n = min(n, len(gr))
            sampled.extend(gr[:n])
    random.shuffle(sampled)
    print(f"  Sampled {len(sampled)} records")
    return sampled


def task3_mini_1000(config):
    print("=" * 60)
    print("TASK 3: mini_1000 Training")
    print("=" * 60)

    # Step 1: Sample data
    print("  [Step 1] Sampling mini_1000 data...")
    sampled = sample_mini_1000()
    MINI1000_DATA.mkdir(parents=True, exist_ok=True)
    write_jsonl(MINI1000_DATA / "train.jsonl", sampled)

    # Copy dev
    import shutil
    shutil.copy2(STAGE3_DATA_DIR / "dev.jsonl", MINI1000_DATA / "validation.jsonl")

    # Compute sha256 and create manifest
    manifest = {}
    for fn in ["train.jsonl", "validation.jsonl"]:
        fp = MINI1000_DATA / fn
        manifest[fn] = {"sha256": sha256_file(fp), "size": fp.stat().st_size}
    # Write dataset_manifest.json (required by training pipeline)
    write_json(MINI1000_DATA / "dataset_manifest.json", {
        "train": {"path": str(MINI1000_DATA / "train.jsonl"), "sha256": manifest["train.jsonl"]["sha256"]},
        "validation": {"path": str(MINI1000_DATA / "validation.jsonl"), "sha256": manifest["validation.jsonl"]["sha256"]},
    })
    print(f"    train: {len(sampled)} records, sha256={manifest['train.jsonl']['sha256'][:16]}...")

    # Step 2: Label audit
    print("  [Step 2] Label audit...")
    from src.training.label_audit import run_label_audit
    mini_config = dict(config)
    mini_config["data"] = dict(config["data"])
    mini_config["data"]["output_dir"] = str(MINI1000_DATA)
    run_label_audit(mini_config)

    batch_audit = json.loads((MINI1000_DATA / "batch_audit.json").read_text())
    token_report = json.loads((MINI1000_DATA / "token_report.json").read_text())
    train_ok = token_report["splits"]["train"]["eligible_count"]
    print(f"    Audit: {batch_audit['status']}, train eligible={train_ok}")

    # Step 3: Training
    print("  [Step 3] Training mini_1000...")
    mini_config["experiment"] = dict(config["experiment"])
    mini_config["experiment"]["output_root"] = str(MINI1000_OUTPUT)
    mini_config["training"] = dict(config["training"])
    mini_config["training"]["smoke_examples"] = len(sampled)  # Use all sampled

    from src.training.train_sft import run_training as sft_run_training
    train_metrics = sft_run_training(mini_config, run_mode="smoke", adapter_path=STAGE2_ADAPTER)

    train_loss = train_metrics.get("train", {}).get("train_loss", "N/A")
    eval_loss = train_metrics.get("validation", {}).get("eval_loss", "N/A")
    steps = train_metrics.get("global_step", "N/A")
    print(f"    Done. Steps={steps}, Train loss={train_loss}, Eval loss={eval_loss}")

    # Step 4: Eval
    print("  [Step 4] Evaluating mini_1000...")
    mini_adapter = str(MINI1000_OUTPUT / "smoke" / "adapter")
    eval_results = run_full_eval(config, mini_adapter, label="mini_1000")

    # Write results
    write_json(MINI1000_OUTPUT / "mini_1000_eval.json", eval_results)
    write_json(MINI1000_OUTPUT / "mini_1000_train_report.json", {
        "config": {"learning_rate": "3e-5", "epochs": 1, "max_length": 1024, "bf16": True,
                   "assistant_only_loss": True, "modules_to_save": ["lm_head"],
                   "init_adapter": str(STAGE2_ADAPTER), "samples": len(sampled)},
        "training": {"steps": steps, "train_loss": train_loss, "eval_loss": eval_loss,
                     "train_runtime": train_metrics.get("train",{}).get("train_runtime"),
                     "peak_gpu_mb": train_metrics.get("peak_gpu_memory_bytes", 0) / 1e6,
                     "supervised_tokens": train_metrics.get("supervised_tokens_seen")},
        "manifest": manifest,
        "eval": eval_results,
    })

    return train_metrics, eval_results


# ═══════════════════════════════════════════════════════════════════════════════
# Final report: side-by-side comparison
# ═══════════════════════════════════════════════════════════════════════════════

def write_final_comparison(baseline, smoke_eval, mini_eval, error_analysis):
    print("=" * 60)
    print("Writing final comparison report...")
    print("=" * 60)

    # Load smoke eval data
    wdev_smoke = smoke_eval.get("wonderland_dev", {})
    wdev_base = baseline.get("wonderland_dev", {})

    lines = [
        "# Stage 3 — Baseline vs Smoke-300 vs mini_1000 — Complete Comparison",
        "",
        "| Metric | Stage 2 (baseline) | Stage 3 Smoke-300 | Stage 3 mini_1000 |",
        "|--------|:------------------:|:-----------------:|:-----------------:|",
    ]

    # Wonderland overall
    base_acc = wdev_base.get("exact_match_rate", 0)
    smoke_acc = wdev_smoke.get("exact_match_rate", 0)
    mini_acc = mini_eval.get("wonderland_dev", {}).get("exact_match_rate", 0) if mini_eval else 0
    lines.append(f"| Wonderland dev exact | {base_acc:.2%} | {smoke_acc:.2%} | {mini_acc:.2%} |")

    # By task type
    for tt in sorted(wdev_base.get("task_type_breakdown", {})):
        ba = wdev_base["task_type_breakdown"].get(tt, {}).get("accuracy", 0)
        sa = wdev_smoke.get("task_type_breakdown", {}).get(tt, {}).get("accuracy", 0)
        ma = mini_eval.get("wonderland_dev", {}).get("task_type_breakdown", {}).get(tt, {}).get("accuracy", 0) if mini_eval else 0
        lines.append(f"| └─ {tt} | {ba:.2%} | {sa:.2%} | {ma:.2%} |")

    # Spurious think
    lines.append(f"| Wonderland spurious think | {wdev_base.get('spurious_think',0)} | {wdev_smoke.get('spurious_think',0)} | {mini_eval.get('wonderland_dev',{}).get('spurious_think',0) if mini_eval else 0} |")
    lines.append(f"| Wonderland missing think | {wdev_base.get('missing_think',0)} | {wdev_smoke.get('missing_think',0)} | {mini_eval.get('wonderland_dev',{}).get('missing_think',0) if mini_eval else 0} |")

    # Final answer parse
    base_parse = wdev_base.get("parse_rate", 0)
    smoke_parse = wdev_smoke.get("parse_rate", 0)
    mini_parse = mini_eval.get("wonderland_dev", {}).get("parse_rate", 0) if mini_eval else 0
    lines.append(f"| Final answer parse rate | {base_parse:.2%} | {smoke_parse:.2%} | {mini_parse:.2%} |")

    # Strict regression
    bs = baseline.get("strict_regression", {})
    ss = smoke_eval.get("strict_regression", {})
    ms = mini_eval.get("strict_regression", {}) if mini_eval else {}
    lines.append(f"| Strict exact match | {bs.get('exact_rate',0):.2%} | {ss.get('exact_rate',0):.2%} | {ms.get('exact_rate',0):.2%} |")
    lines.append(f"| Strict spurious think | {bs.get('spurious_rate',0):.2%} | {ss.get('spurious_rate',0):.2%} | {ms.get('spurious_rate',0):.2%} |")

    # Protocol regression
    bp = baseline.get("protocol_regression", {})
    sp = smoke_eval.get("protocol_regression", {})
    mp = mini_eval.get("protocol_regression", {}) if mini_eval else {}
    lines.append(f"| Protocol think both tags | {bp.get('thinking',{}).get('rate',0):.2%} | {sp.get('thinking',{}).get('rate',0):.2%} | {mp.get('thinking',{}).get('rate',0):.2%} |")
    lines.append(f"| Protocol no-think clean | {bp.get('no_thinking',{}).get('rate',0):.2%} | {sp.get('no_thinking',{}).get('rate',0):.2%} | {mp.get('no_thinking',{}).get('rate',0):.2%} |")

    # Open regression
    bo = baseline.get("open_regression", {})
    so = smoke_eval.get("open_regression", {})
    mo = mini_eval.get("open_regression", {}) if mini_eval else {}
    lines.append(f"| Open too-short rate | {bo.get('too_short_rate',0):.2%} | {so.get('too_short_rate',0):.2%} | {mo.get('too_short_rate',0):.2%} |")
    lines.append(f"| Open spurious think | {bo.get('spurious_rate',0):.2%} | {so.get('spurious_rate',0):.2%} | {mo.get('spurious_rate',0):.2%} |")
    lines.append(f"| Open mean words | {bo.get('word_mean',0)} | {so.get('word_mean',0)} | {mo.get('word_mean',0)} |")

    lines.append("")
    lines.append("## Error Analysis Summary")
    lines.append("")
    if error_analysis:
        for tt, data in error_analysis.items():
            classes = Counter(s["error_class"] for s in data["samples"])
            top_class = classes.most_common(1)[0][0] if classes else "N/A"
            lines.append(f"- **{tt}**: {data['total_errors']} errors, top cause: `{top_class}` "
                         f"({', '.join(f'{k}={v}' for k,v in classes.most_common(3))})")

    out_path = Path("outputs/stage3_wonderland_coldstart/comparison_report.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Written: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Stage 3 baseline eval + error analysis + mini_1000")
    p.add_argument("--config", type=Path, default=Path("configs/stage3_wonderland_coldstart.yaml"))
    p.add_argument("--task", choices=["all", "baseline", "error", "mini1000", "compare"], default="all")
    p.add_argument("--set", action="append", default=[], dest="overrides")
    return p.parse_args()


def main():
    args = parse_args()
    config = load_yaml_config(args.config)
    if args.overrides:
        config = apply_overrides(config, args.overrides)

    baseline = None
    error_analysis = None
    mini_train = None
    mini_eval = None
    smoke_eval = None

    if args.task in ("all", "baseline"):
        baseline = task1_baseline_eval(config)

    if args.task in ("all", "error"):
        error_analysis = task2_error_analysis(config)

    if args.task in ("all", "mini1000"):
        mini_train, mini_eval = task3_mini_1000(config)

    if args.task in ("all", "compare"):
        # Load smoke eval
        smoke_path = Path("outputs/stage3_wonderland_coldstart/smoke_300")
        # We need the full eval data. Let's just run it if not loaded.
        if not smoke_eval:
            smoke_eval = run_full_eval(config, str(SMOKE_ADAPTER), label="smoke_300")
        if not baseline:
            bp = BASELINE_OUTPUT / "baseline_eval.json"
            if bp.is_file():
                baseline = json.loads(bp.read_text())
            else:
                baseline = task1_baseline_eval(config)
        if not mini_eval:
            mp = MINI1000_OUTPUT / "mini_1000_eval.json"
            if mp.is_file():
                mini_eval = json.loads(mp.read_text())
        if not error_analysis:
            ep = Path("outputs/stage3_wonderland_coldstart/smoke_error_analysis.json")
            if ep.is_file():
                error_analysis = json.loads(ep.read_text())

        write_final_comparison(baseline, smoke_eval, mini_eval, error_analysis)

    print("\nDone.")


if __name__ == "__main__":
    main()
