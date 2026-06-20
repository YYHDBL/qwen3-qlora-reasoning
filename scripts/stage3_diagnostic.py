#!/usr/bin/env python3
"""Stage 3 diagnostic — Task 1 & 2:
  1. Stage 2 vs Stage 3 formal_v0_1 side-by-side on wonderland-only 116 dev
  2. formal_v0_1 error decomposition by task_type + error class
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.config import load_yaml_config
from src.common.experiment import write_json, sha256_file

STAGE2_ADAPTER = Path("outputs/stage2_thinking_warmup/formal/adapter")
STAGE3_FORMAL_ADAPTER = Path("outputs/stage3_wonderland_coldstart/formal/adapter")
STAGE3_DEV = Path("data/instruction/stage3_wonderland_cold_start/dev.jsonl")
OUT_DIR = Path("outputs/stage3_wonderland_coldstart/diagnostic")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_model_eval(config, adapter_path):
    """Reuse same loader as formal_v0_1 script."""
    import torch
    from src.training.model_loader import load_lora_model, load_tokenizer
    from src.training.chat_template import resolve_stop_token_ids
    tokenizer = load_tokenizer(config, for_training=False)
    model = load_lora_model(config, adapter_path, is_trainable=False).eval()
    model.config.use_cache = True
    stop_ids = resolve_stop_token_ids(tokenizer)
    return model, tokenizer, stop_ids


def _gen(model, tokenizer, stop_ids, msgs, max_tok=256, thinking=None):
    import torch
    from src.training.chat_template import render_generation_prompt
    p = render_generation_prompt(tokenizer, list(msgs), enable_thinking=thinking)
    inp = tokenizer([p], return_tensors="pt").to(model.device)
    n = inp.input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_tok, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id,
                             eos_token_id=[tokenizer.eos_token_id, stop_ids.get("im_end"), stop_ids.get("endoftext")])
    return tokenizer.decode(out[0][n:], skip_special_tokens=True).strip()


def evaluate(adapter_path, label, wonderland_dev, config):
    """Run eval and return per-sample results + summary."""
    import torch
    model, tok, sid = _load_model_eval(config, adapter_path)
    per = []
    by_task = defaultdict(lambda: {"correct": 0, "total": 0, "tokens_sum": 0})

    for rec in wonderland_dev:
        tt = rec["task_type"]
        st = rec["sample_type"]
        user_msgs = [m for m in rec["messages"] if m["role"] == "user"]
        expected = rec["messages"][-1]["content"].strip()
        wants_think = st == "wonderland_compressed_cot"
        gen = _gen(model, tok, sid, user_msgs, max_tok=256, thinking=wants_think)
        has_open = "<think>" in gen
        has_close = "</think>" in gen
        final_answer = gen.split("</think>")[-1].strip() if has_close else gen.strip()
        tc = len(tok.encode(gen))
        expected_n = " ".join(expected.split())
        final_n = " ".join(final_answer.split())
        ok = final_n == expected_n

        by_task[tt]["total"] += 1
        by_task[tt]["tokens_sum"] += tc
        if ok:
            by_task[tt]["correct"] += 1

        per.append({
            "id": rec["id"],
            "task": tt,
            "sample": st,
            "exact": ok,
            "think_open": has_open,
            "think_close": has_close,
            "tokens": tc,
            "expected": expected,
            "final": final_answer,
            "raw_gen": gen if not ok else "",
        })

    del model
    torch.cuda.empty_cache()

    total = len(wonderland_dev)
    exact_count = sum(1 for r in per if r["exact"])
    cot_samples = [r for r in per if r["sample"] == "wonderland_compressed_cot"]
    parse_ok = sum(1 for r in cot_samples if r["think_close"] and r["final"])
    parse_total = len(cot_samples)
    stop_ok = sum(1 for r in per if r["tokens"] < 256)
    mean_tokens = sum(r["tokens"] for r in per) / total if total else 0

    summary = {
        "label": label,
        "adapter": str(adapter_path),
        "total": total,
        "exact": exact_count,
        "exact_rate": round(exact_count / total, 4) if total else 0,
        "parse_ok": parse_ok,
        "parse_total": parse_total,
        "parse_rate": round(parse_ok / parse_total, 4) if parse_total else 0,
        "stop_ok": stop_ok,
        "stop_rate": round(stop_ok / total, 4) if total else 0,
        "mean_tokens": round(mean_tokens, 1),
        "by_task": {
            tt: {
                "correct": c["correct"], "total": c["total"],
                "accuracy": round(c["correct"] / c["total"], 4) if c["total"] else 0,
                "mean_tokens": round(c["tokens_sum"] / c["total"], 1) if c["total"] else 0,
            }
            for tt, c in sorted(by_task.items())
        },
    }
    return per, summary


def decompose_errors(predictions):
    """Task 2: error decomposition by error class."""
    import re
    import decimal
    from decimal import Decimal

    errors = [r for r in predictions if not r["exact"]]
    by_task = defaultdict(list)

    for e in errors:
        tt = e["task"]
        exp = e["expected"].strip()
        fin = e["final"].strip()
        raw = e.get("raw_gen", "")

        # Determine error class
        if not e["think_close"] and e["sample"] == "wonderland_compressed_cot":
            ec = "parse_fail"
        elif "<think>" in raw and "</think>" not in raw and len(raw) > 300:
            ec = "loop"
        elif not fin:
            ec = "parse_fail"
        elif tt in ("gravity", "unit_conversion"):
            ec = _classify_numeric_error(raw, exp, fin, tt)
        elif tt == "bit_manipulation":
            ec = _classify_bit_error(raw, exp, fin)
        elif tt == "cipher":
            ec = "mapping_wrong"
        elif tt == "numeral":
            ec = "rule_wrong"
        elif tt == "symbolic_equation":
            ec = "rule_wrong"
        else:
            ec = "final_mismatch"

        e["error_class"] = ec
        by_task[tt].append(e)

    # Gravity/unit special analysis
    grav_unit_errors = [e for e in errors if e["task"] in ("gravity", "unit_conversion")]
    gu_analysis = _analyze_gravity_unit_errors(grav_unit_errors)

    # Error class counts
    error_classes = defaultdict(int)
    for e in errors:
        error_classes[e["error_class"]] += 1

    return {
        "total_errors": len(errors),
        "error_class_counts": dict(sorted(error_classes.items(), key=lambda x: -x[1])),
        "by_task_type": {
            tt: {
                "count": len(elist),
                "error_class_breakdown": _count_classes(elist),
                "samples": [
                    {"id": e["id"], "task": e["task"], "sample": e["sample"],
                     "error_class": e["error_class"], "expected": e["expected"],
                     "final": e["final"], "raw_gen": e.get("raw_gen","")[:200]}
                    for e in elist[:5]
                ],
            }
            for tt, elist in sorted(by_task.items())
        },
        "gravity_unit_analysis": gu_analysis,
    }


def _count_classes(error_list):
    ec = defaultdict(int)
    for e in error_list:
        ec[e["error_class"]] += 1
    return dict(sorted(ec.items(), key=lambda x: -x[1]))


def _classify_numeric_error(raw, exp, fin, tt):
    import re
    from decimal import Decimal, InvalidOperation
    try:
        gold = Decimal(exp)
        pred = Decimal(fin)
    except (InvalidOperation, ValueError):
        return "parse_fail"

    if gold == 0:
        return "final_mismatch"

    diff = abs(gold - pred)
    if diff == 0:
        return "final_mismatch"

    # Close → rounding
    if float(diff / gold) < 0.01:
        return "rounding_wrong"

    # Try to detect coefficient error: ratio near 1 but not 1
    ratio = float(pred / gold)
    if 0.7 < ratio < 1.3 and abs(ratio - 1.0) > 0.01:
        return "coefficient_wrong"

    # Check raw generation for formula hints
    if raw:
        has_formula = bool(re.search(r"g\s*[=≈]|ratio|coefficient|2\*d|0\.5\*g|output/input", raw, re.IGNORECASE))
        has_computation = bool(re.search(r"\d+\.?\d*\s*[*×]\s*\d+\.?\d*|0\.5\s*[*×]", raw))
        if has_formula:
            return "arithmetic_wrong" if has_computation else "formula_wrong"

    return "arithmetic_wrong"


def _classify_bit_error(raw, exp, fin):
    if not fin:
        return "parse_fail"
    if len(fin) != len(exp):
        return "mapping_wrong"
    diffs = sum(1 for a, b in zip(fin, exp) if a != b)
    return "mapping_wrong" if diffs > 0 else "final_mismatch"


def _analyze_gravity_unit_errors(errors):
    import re
    results = []
    for e in errors:
        raw = e.get("raw_gen", "")
        tt = e["task"]
        a = {"id": e["id"], "task": tt, "gold": e["expected"],
             "parsed": e["final"], "error_class": e["error_class"],
             "raw_preview": raw[:300]}

        if tt == "gravity":
            a["formula_correct"] = bool(re.search(r"g\s*[=≈]|2\*d/\s*t\^2|0\.5\*g\*t\^2", raw, re.IGNORECASE))
            a["coefficient_extracted"] = bool(re.search(r"g\s*[≈=]\s*\d+\.?\d*", raw))
            a["computation_present"] = bool(re.search(r"0\.5\s*[*×]\s*\d+\.?\d*\s*[*×]", raw))
        elif tt == "unit_conversion":
            a["formula_correct"] = bool(re.search(r"ratio|coefficient|output/input", raw, re.IGNORECASE))
            a["coefficient_extracted"] = bool(re.search(r"ratio|coefficient.*?\d+\.\d+", raw, re.IGNORECASE))
            a["computation_present"] = bool(re.search(r"\*\s*coefficient|coefficient\s*\*|multiply|×", raw, re.IGNORECASE))

        results.append(a)

    stats = {
        "total": len(errors),
        "formula_correct": sum(1 for r in results if r.get("formula_correct")),
        "coefficient_extracted": sum(1 for r in results if r.get("coefficient_extracted")),
        "computation_present": sum(1 for r in results if r.get("computation_present")),
    }
    return {"stats": stats, "samples": results}


def print_side_by_side(s2, s3):
    print()
    print("=" * 80)
    print(" Stage 2 vs Stage 3 formal_v0_1 — Wonderland-only 116 dev")
    print("=" * 80)
    print(f"{'Metric':<25} {'Stage 2 baseline':>18} {'Stage 3 formal_v0_1':>20}")
    print("-" * 65)
    for label, key in [("overall_exact_rate", "exact_rate"), ("parse_rate", "parse_rate"),
                        ("stop_rate", "stop_rate"), ("mean_tokens", "mean_tokens")]:
        print(f"{label:<25} {s2[key]:>18.4f} {s3[key]:>20.4f}")

    print()
    print(f"{'Task Type':<22} {'S2 Acc':>8} {'S2 Toks':>8} {'S3 Acc':>8} {'S3 Toks':>8} {'Delta':>8}")
    print("-" * 60)
    all_tt = sorted(set(list(s2["by_task"].keys()) + list(s3["by_task"].keys())))
    for tt in all_tt:
        s2t = s2["by_task"].get(tt, {"accuracy": 0, "mean_tokens": 0})
        s3t = s3["by_task"].get(tt, {"accuracy": 0, "mean_tokens": 0})
        print(f"{tt:<22} {s2t['accuracy']:>8.4f} {s2t['mean_tokens']:>7.1f} {s3t['accuracy']:>8.4f} {s3t['mean_tokens']:>7.1f} {s3t['accuracy']-s2t['accuracy']:>+8.4f}")

    print()
    print(f"  S2 sum: exact={s2['exact']}/{s2['total']}")
    print(f"  S3 sum: exact={s3['exact']}/{s3['total']}")

    delta = s3["exact_rate"] - s2["exact_rate"]
    print()
    if abs(delta) < 0.02:
        print(f"  CONCLUSION: Stage 3 ≈ Stage 2 (Δ={delta:+.4f})")
        print(f"  Model did not degrade, but also did not learn Wonderland tasks.")
    elif delta < -0.02:
        print(f"  CONCLUSION: Stage 3 < Stage 2 (Δ={delta:+.4f})")
        print(f"  Training HARMED accuracy — data quality/ratio issue.")
    else:
        print(f"  CONCLUSION: Stage 3 > Stage 2 (Δ={delta:+.4f})")
        print(f"  Some learning occurred but minimal.")


def main():
    config_path = Path("configs/stage3_wonderland_coldstart.yaml")
    config = load_yaml_config(config_path)

    # Load wonderland dev
    all_dev = [json.loads(l) for l in open(STAGE3_DEV) if l.strip()]
    wonderland_dev = [r for r in all_dev if r["sample_type"] in ("wonderland_answer_only", "wonderland_compressed_cot")]
    print(f"[diag] Wonderland dev: {len(wonderland_dev)} samples")

    # ── Task 1: Stage 2 baseline ──
    print("[diag] Running Stage 2 baseline...")
    s2_per, s2_summary = evaluate(STAGE2_ADAPTER, "Stage 2 baseline", wonderland_dev, config)

    # ── Task 1: Stage 3 formal_v0_1 ──
    print("[diag] Running Stage 3 formal_v0_1...")
    s3_per, s3_summary = evaluate(STAGE3_FORMAL_ADAPTER, "Stage 3 formal_v0_1", wonderland_dev, config)

    # ── Side-by-side ──
    print_side_by_side(s2_summary, s3_summary)
    comparison = {"stage2_baseline": s2_summary, "stage3_formal_v0_1": s3_summary}
    write_json(OUT_DIR / "baseline_comparison.json", comparison)
    print(f"\n[diag] comparison saved to {OUT_DIR / 'baseline_comparison.json'}")

    # ── Task 2: Error decomposition ──
    print("\n" + "=" * 80)
    print(" Task 2: formal_v0_1 Error Decomposition")
    print("=" * 80)
    error_report = decompose_errors(s3_per)

    print(f"\nTotal errors: {error_report['total_errors']}")
    print(f"\nError class breakdown:")
    for ec, cnt in error_report["error_class_counts"].items():
        pct = cnt / error_report["total_errors"] * 100 if error_report["total_errors"] else 0
        print(f"  {ec:<25} {cnt:>4} ({pct:>5.1f}%)")

    print(f"\nPer-task breakdown:")
    for tt, info in error_report["by_task_type"].items():
        print(f"  {tt}: {info['count']} errors — {info['error_class_breakdown']}")

    print(f"\nGravity/Unit analysis:")
    gu = error_report["gravity_unit_analysis"]
    s = gu["stats"]
    print(f"  Error count: {s['total']}")
    print(f"  Formula correct: {s['formula_correct']}")
    print(f"  Coefficient extracted: {s['coefficient_extracted']}")
    print(f"  Computation present: {s['computation_present']}")
    print(f"\n  Sample errors:")
    for r in gu["samples"][:5]:
        print(f"    {r['id']} ({r['task']}): gold={r['gold']} parsed={r['parsed']} [{r['error_class']}]")
        for k in ["formula_correct", "coefficient_extracted", "computation_present"]:
            if k in r:
                print(f"      {k}: {r[k]}")

    write_json(OUT_DIR / "error_decomposition.json", error_report)
    write_json(OUT_DIR / "s3_predictions.json", s3_per)
    print(f"\n[diag] All results saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
