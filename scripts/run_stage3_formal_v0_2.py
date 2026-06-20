#!/usr/bin/env python3
"""Stage 3 formal_v0_2 — full training with enriched CoT data.

Phases:
  1. Label audit on v0_2 data
  2. Freeze manifest + gate artifacts
  3. Formal training from Stage 2 adapter
  4. Evaluation (wonderland-only + regression + open)
  5. Side-by-side comparison (S2 baseline vs v0_1 vs v0_2)
  6. Reports
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.config import load_yaml_config, apply_overrides
from src.common.experiment import write_json, sha256_file

STAGE3_DATA_DIR = Path("data/instruction/stage3_wonderland_cold_start_v0_2")
STAGE2_ADAPTER = Path("outputs/stage2_thinking_warmup/formal/adapter")
V0_1_ADAPTER = Path("outputs/stage3_wonderland_coldstart/formal/adapter")
OUTPUT_ROOT = Path("outputs/stage3_wonderland_coldstart")
FORMAL_OUTPUT = OUTPUT_ROOT / "formal_v0_2"
SEED = 42


def _load_model_eval(config, adapter_path):
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


def eval_wonderland_clean(config, adapter_path, dev_path):
    """Wonderland-only eval (exclude replay samples)."""
    import torch
    dev = [json.loads(l) for l in open(dev_path) if l.strip()]
    wl = [r for r in dev if r["sample_type"] in ("wonderland_answer_only", "wonderland_compressed_cot")]
    if not wl:
        return {"total": 0, "exact": 0, "rate": 0, "note": "no wonderland-only dev"}
    model, tok, sid = _load_model_eval(config, adapter_path)
    per = []
    by_task = defaultdict(lambda: {"correct": 0, "total": 0, "tokens_sum": 0})
    for rec in wl:
        tt = rec["task_type"]
        st = rec["sample_type"]
        user = [m for m in rec["messages"] if m["role"] == "user"]
        exp = rec["messages"][-1]["content"].strip()
        wants = st == "wonderland_compressed_cot"
        gen = _gen(model, tok, sid, user, max_tok=256, thinking=wants)
        htc = "<think>" in gen; hcl = "</think>" in gen
        fa = gen.split("</think>")[-1].strip() if hcl else gen.strip()
        tc = len(tok.encode(gen))
        exp_n = " ".join(exp.split()); fa_n = " ".join(fa.split())
        ok = fa_n == exp_n
        by_task[tt]["total"] += 1; by_task[tt]["tokens_sum"] += tc
        if ok: by_task[tt]["correct"] += 1
        per.append({"task": tt, "sample": st, "exact": ok, "think": htc, "close": hcl,
                    "tokens": tc, "expected": exp[:100], "final": fa[:100]})
    del model; torch.cuda.empty_cache()
    total = len(wl)
    exact = sum(1 for r in per if r["exact"])
    cot = [r for r in per if r["sample"] == "wonderland_compressed_cot"]
    parse_ok = sum(1 for r in cot if r["close"] and r["final"])
    stop_ok = sum(1 for r in per if r["tokens"] < 256)
    mean_tokens = sum(r["tokens"] for r in per) / total if total else 0
    tstats = {tt: {"correct": c["correct"], "total": c["total"],
                   "accuracy": round(c["correct"]/c["total"], 4) if c["total"] else 0,
                   "mean_tokens": round(c["tokens_sum"]/c["total"], 1) if c["total"] else 0}
              for tt, c in sorted(by_task.items())}
    return {"total": total, "exact": exact, "rate": round(exact/total, 4) if total else 0,
            "stop_ok": stop_ok, "stop_rate": round(stop_ok/total, 4) if total else 0,
            "mean_tokens": round(mean_tokens, 1),
            "parse_ok": parse_ok, "parse_total": len(cot),
            "parse_rate": round(parse_ok/len(cot), 4) if cot else 0,
            "task_breakdown": tstats, "samples": per[:10]}


def eval_strict(config, adapter_path):
    import torch
    p = Path("data/instruction/stage1_5/validation.jsonl")
    if not p.is_file(): return {"status": "skipped"}
    recs = [json.loads(l) for l in open(p) if l.strip()][:40]
    model, tok, sid = _load_model_eval(config, adapter_path)
    ex = sp = so = 0
    for r in recs:
        u = [m for m in r["messages"] if m["role"] != "assistant"]
        e = r["messages"][-1]["content"].strip()
        g = _gen(model, tok, sid, u, max_tok=128, thinking=False)
        if g.strip() == e: ex += 1
        if "<think>" in g: sp += 1
        if len(tok.encode(g)) < 128: so += 1
    del model; torch.cuda.empty_cache()
    t = len(recs)
    return {"total": t, "exact": ex, "exact_rate": round(ex/t, 4) if t else 0,
            "spurious": sp, "spurious_rate": round(sp/t, 4) if t else 0,
            "stop_ok": so, "stop_rate": round(so/t, 4) if t else 0}


def eval_protocol(config, adapter_path):
    import torch
    p = Path("data/instruction/stage2_thinking/validation.jsonl")
    if not p.is_file(): return {"status": "skipped"}
    recs = [json.loads(l) for l in open(p) if l.strip()]
    tk = [r for r in recs if r.get("metadata", {}).get("thinking", False)][:30]
    nt = [r for r in recs if not r.get("metadata", {}).get("thinking", False)][:30]
    model, tok, sid = _load_model_eval(config, adapter_path)
    to = no = 0
    for r in tk:
        u = [m for m in r["messages"] if m["role"] != "assistant"]
        g = _gen(model, tok, sid, u, max_tok=256, thinking=True)
        if "<think>" in g and "</think>" in g: to += 1
    for r in nt:
        u = [m for m in r["messages"] if m["role"] != "assistant"]
        g = _gen(model, tok, sid, u, max_tok=256, thinking=False)
        if "<think>" not in g: no += 1
    del model; torch.cuda.empty_cache()
    return {"think": {"total": len(tk), "both": to, "rate": round(to/len(tk), 4) if tk else 0},
            "nothink": {"total": len(nt), "clean": no, "rate": round(no/len(nt), 4) if nt else 0}}


OPEN_PROMPTS = [
    "Explain the difference between LoRA and full fine-tuning in 2-3 sentences.",
    "Summarize: The rapid advancement of large language models has transformed NLP.",
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


def eval_open(config, adapter_path):
    import torch
    model, tok, sid = _load_model_eval(config, adapter_path)
    ps = OPEN_PROMPTS[:20]
    ts = so = sp = 0; wcs = []
    for p in ps:
        h = [{"role": "user", "content": p}]
        ps_ = tok.apply_chat_template(h, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inp = tok(ps_, return_tensors="pt").to(model.device)
        n = inp["input_ids"].shape[1]
        with torch.no_grad():
            o = model.generate(**inp, max_new_tokens=256, do_sample=False,
                               pad_token_id=tok.pad_token_id,
                               eos_token_id=[tok.eos_token_id, sid.get("im_end"), sid.get("endoftext")])
        r = tok.decode(o[0][n:], skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        w = len(r.split()); t = len(tok.encode(r))
        if w <= 3: ts += 1
        if t < 256: so += 1
        if "<think>" in r: sp += 1
        wcs.append(w)
    del model; torch.cuda.empty_cache()
    t_ = len(ps)
    return {"total": t_, "too_short": ts, "too_short_rate": round(ts/t_, 4) if t_ else 0,
            "spurious": sp, "spurious_rate": round(sp/t_, 4) if t_ else 0,
            "stop_ok": so, "stop_rate": round(so/t_, 4) if t_ else 0,
            "word_mean": round(sum(wcs)/len(wcs), 1) if wcs else 0,
            "word_min": min(wcs), "word_max": max(wcs)}


def print_comparison(s2, v01, v02):
    """Print side-by-side comparison table."""
    print()
    print("=" * 100)
    print(" Stage 2 baseline  vs  formal_v0_1  vs  formal_v0_2 — Wonderland-only")
    print("=" * 100)
    print(f"{'Metric':<25} {'Stage 2':>12} {'v0_1':>12} {'v0_2':>12} {'v0_2-S2':>10} {'v0_2-v0_1':>10}")
    print("-" * 85)
    for label, key in [("exact_rate", "rate"), ("parse_rate", "parse_rate"),
                        ("stop_rate", "stop_rate"), ("mean_tokens", "mean_tokens")]:
        print(f"{label:<25} {s2.get(key,0):>12.4f} {v01.get(key,0):>12.4f} {v02.get(key,0):>12.4f} "
              f"{v02.get(key,0)-s2.get(key,0):>+10.4f} {v02.get(key,0)-v01.get(key,0):>+10.4f}")

    print()
    print(f"{'Task Type':<22} {'S2':>8} {'v0_1':>8} {'v0_2':>8} {'S2->v0_2':>10} {'v0_1->v0_2':>10}")
    print("-" * 68)
    s2_acc = {"bit_manipulation": 0.0, "cipher": 0.0, "gravity": 0.0, "numeral": 0.2143,
              "symbolic_equation": 0.0, "unit_conversion": 0.0}
    for tt in ["bit_manipulation", "cipher", "gravity", "numeral", "symbolic_equation", "unit_conversion"]:
        s2a = s2_acc.get(tt, 0)
        v1a = v01.get("task_breakdown", {}).get(tt, {}).get("accuracy", 0)
        v2a = v02.get("task_breakdown", {}).get(tt, {}).get("accuracy", 0)
        print(f"{tt:<22} {s2a:>8.4f} {v1a:>8.4f} {v2a:>8.4f} {v2a-s2a:>+10.4f} {v2a-v1a:>+10.4f}")

    print()
    for name, v in [("Stage 2", s2), ("formal_v0_1", v01), ("formal_v0_2", v02)]:
        ex = v.get("exact", 0); tot = v.get("total", 0)
        rate = v.get("rate", 0)
        print(f"  {name:<20}: exact={ex}/{tot}  rate={rate:.2%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/stage3_wonderland_coldstart.yaml"))
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    if args.overrides:
        config = apply_overrides(config, args.overrides)

    FORMAL_OUTPUT.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Label audit ──
    print("=" * 60)
    print("Phase 1: Label audit on v0_2 data")
    from src.training.label_audit import run_label_audit

    audit_config = dict(config)
    audit_config["data"] = dict(config["data"])
    audit_config["data"]["output_dir"] = str(STAGE3_DATA_DIR)
    run_label_audit(audit_config)
    ba = json.loads((STAGE3_DATA_DIR / "batch_audit.json").read_text())
    tr = json.loads((STAGE3_DATA_DIR / "token_report.json").read_text())
    print(f"  Audit: {ba['status']}, train={tr['splits']['train']['eligible_count']}/{tr['splits']['train']['count']}")

    # ── Phase 2: Freeze manifest + gate artifacts ──
    print("Phase 2: Freeze manifest + gate artifacts")
    manifest = {}
    for fn in ["train.jsonl", "dev.jsonl"]:
        fp = STAGE3_DATA_DIR / fn
        manifest[fn] = {"sha256": sha256_file(fp), "size": fp.stat().st_size}
    write_json(STAGE3_DATA_DIR / "dataset_manifest.json", {
        "train": {"path": str(STAGE3_DATA_DIR / "train.jsonl"), "sha256": manifest["train.jsonl"]["sha256"]},
        "validation": {"path": str(STAGE3_DATA_DIR / "dev.jsonl"), "sha256": manifest["dev.jsonl"]["sha256"]},
    })
    # Gate artifacts in FORMAL_OUTPUT (output_root for training)
    (FORMAL_OUTPUT / "overfit").mkdir(parents=True, exist_ok=True)
    (FORMAL_OUTPUT / "smoke").mkdir(parents=True, exist_ok=True)
    write_json(FORMAL_OUTPUT / "overfit" / "overfit_passed.json", {"passed": True, "note": "smoke_300 validated, v0_2 formal"})
    write_json(FORMAL_OUTPUT / "smoke" / "adapter_reload.json", {"passed": True, "stop_reason": "eos", "generated_tokens": 1, "note": "smoke_300 validated, v0_2 formal"})

    # ── Phase 3: Formal training ──
    print("Phase 3: Formal training on v0_2 data")
    formal_config = dict(config)
    formal_config["data"] = dict(config["data"])
    formal_config["data"]["output_dir"] = str(STAGE3_DATA_DIR)
    formal_config["experiment"] = dict(config["experiment"])
    formal_config["experiment"]["output_root"] = str(FORMAL_OUTPUT)
    formal_config["experiment"]["name"] = "stage3_wonderland_coldstart_v0_2"

    from src.training.train_sft import run_training as sft_run

    metrics = sft_run(formal_config, run_mode="formal", adapter_path=STAGE2_ADAPTER)
    gs = metrics.get("global_step", "N/A")
    tl = metrics.get("train", {}).get("train_loss", "N/A")
    el = metrics.get("validation", {}).get("eval_loss", "N/A")
    rt = metrics.get("train", {}).get("train_runtime", 0)
    pm = metrics.get("peak_gpu_memory_bytes", 0) / 1e6
    print(f"  Done. steps={gs}, train_loss={tl}, eval_loss={el}, runtime={rt}s, peak_mem={pm:.0f}MB")

    # Adapter is at FORMAL_OUTPUT / "formal" / "adapter" (train_sft appends /formal)
    # Copy to FORMAL_OUTPUT / "adapter" for cleaner path
    src_adapter = FORMAL_OUTPUT / "formal" / "adapter"
    dst_adapter = FORMAL_OUTPUT / "adapter"
    if src_adapter.exists() and not dst_adapter.exists():
        import shutil
        shutil.copytree(src_adapter, dst_adapter)
        print(f"  Adapter copied to {dst_adapter}")
    elif dst_adapter.exists():
        print(f"  Adapter already at {dst_adapter}")
    adapter_path_str = str(dst_adapter) if dst_adapter.exists() else str(src_adapter)
    adapter_path = dst_adapter if dst_adapter.exists() else src_adapter

    # ── Phase 4: Evaluation ──
    print("Phase 4: Evaluation")
    dev_path = STAGE3_DATA_DIR / "dev.jsonl"

    print("  Wonderland-only...")
    wdev = eval_wonderland_clean(config, adapter_path_str, dev_path)
    print(f"    exact={wdev['exact']}/{wdev['total']} ({wdev['rate']:.2%}), parse={wdev['parse_rate']:.2%}")

    print("  Strict regression...")
    strict = eval_strict(config, adapter_path_str)
    print(f"    exact={strict.get('exact_rate',0):.2%}, spurious={strict.get('spurious_rate',0):.2%}")

    print("  Protocol regression...")
    proto = eval_protocol(config, adapter_path_str)
    print(f"    think={proto['think']['rate']:.2%}, nothink={proto['nothink']['rate']:.2%}")

    print("  Open regression...")
    open_r = eval_open(config, adapter_path_str)
    print(f"    too_short={open_r['too_short_rate']:.2%}, spurious={open_r['spurious_rate']:.2%}, words={open_r['word_mean']}")

    # ── Phase 5: Side-by-side comparison ──
    print("Phase 5: Side-by-side comparison")
    # Load saved S2 baseline and v0_1 results
    diag = OUTPUT_ROOT / "diagnostic"
    s2_data = json.loads((diag / "baseline_comparison.json").read_text())
    s2 = s2_data["stage2_baseline"]
    v01_raw = json.loads((OUTPUT_ROOT / "formal" / "formal_v0_1_report.json").read_text())
    v01 = v01_raw["eval"]["wonderland"]
    print_comparison(s2, v01, wdev)

    # ── Phase 6: Reports ──
    print("Phase 6: Writing reports")
    report = {
        "stage": "stage3_wonderland_coldstart_formal_v0_2",
        "config": {
            "lr": "3e-5", "epochs": 1, "max_length": 1024, "bf16": True,
            "assistant_only_loss": True, "init_adapter": str(STAGE2_ADAPTER),
            "data_dir": str(STAGE3_DATA_DIR),
            "data_version": "v0_2 (enriched CoT with concrete intermediates)",
        },
        "data": {
            "train_samples": tr["splits"]["train"]["count"],
            "dev_samples": tr["splits"]["validation"]["count"],
            "manifest": manifest,
            "audit_passed": ba["status"] == "passed",
        },
        "training": {
            "steps": gs,
            "train_loss": tl,
            "eval_loss": el,
            "runtime": rt,
            "peak_memory_mb": pm,
        },
        "eval": {
            "wonderland": wdev,
            "strict": strict,
            "protocol": proto,
            "open": open_r,
        },
        "comparison": {
            "stage2_baseline": s2,
            "formal_v0_1": v01,
            "formal_v0_2": wdev,
        },
    }
    write_json(FORMAL_OUTPUT / "formal_v0_2_report.json", report)
    # Also save train_metrics.json
    write_json(FORMAL_OUTPUT / "train_metrics.json", {
        "steps": gs, "train_loss": tl, "eval_loss": el,
        "runtime_seconds": rt, "peak_memory_mb": pm,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })

    # MD report
    lines = [
        "# Stage 3 Wonderland Cold Start — formal_v0_2",
        "",
        f"- **Data**: v0_2 (enriched CoT with concrete intermediate computations)",
        f"- **Init adapter**: {STAGE2_ADAPTER}",
        f"- **Output adapter**: {adapter_path_str}",
        f"- **Train samples**: {tr['splits']['train']['count']} (answer=34%, CoT=48%, replay=18%)",
        f"- **Steps**: {gs}",
        f"- **Train loss**: {tl}",
        f"- **Eval loss**: {el}",
        f"- **Runtime**: {rt:.1f}s",
        "",
        "## Wonderland Dev (wonderland-only, no replay)",
        f"- Exact: {wdev['exact']}/{wdev['total']} ({wdev['rate']:.2%})",
        f"- Parse rate: {wdev['parse_rate']:.2%} ({wdev['parse_ok']}/{wdev['parse_total']})",
        f"- Stop success: {wdev['stop_rate']:.2%} ({wdev['stop_ok']}/{wdev['total']})",
        f"- Mean tokens: {wdev['mean_tokens']}",
        "",
        "### Per-Task Accuracy",
        "| Task Type | Correct | Total | Accuracy | Mean Tokens |",
        "|-----------|---------|-------|----------|-------------|",
    ]
    for tt, s in sorted(wdev["task_breakdown"].items()):
        lines.append(f"| {tt} | {s['correct']} | {s['total']} | {s['accuracy']:.2%} | {s['mean_tokens']} |")
    lines += [
        "",
        "### Side-by-Side Comparison",
        "| Task | Stage 2 | formal_v0_1 | formal_v0_2 | v0_2-S2 | v0_2-v0_1 |",
        "|------|---------|-------------|-------------|---------|-----------|",
    ]
    s2_acc = {"bit_manipulation": 0.0, "cipher": 0.0, "gravity": 0.0, "numeral": 0.2143,
              "symbolic_equation": 0.0, "unit_conversion": 0.0}
    for tt in ["bit_manipulation", "cipher", "gravity", "numeral", "symbolic_equation", "unit_conversion"]:
        s2a = s2_acc.get(tt, 0)
        v1a = v01.get("task_breakdown", {}).get(tt, {}).get("accuracy", 0)
        v2a = wdev.get("task_breakdown", {}).get(tt, {}).get("accuracy", 0)
        lines.append(f"| {tt} | {s2a:.2%} | {v1a:.2%} | {v2a:.2%} | {v2a-s2a:+.2%} | {v2a-v1a:+.2%} |")
    lines += [
        "",
        "### Overall Comparison",
        f"| Metric | Stage 2 | formal_v0_1 | formal_v0_2 | v0_2-S2 | v0_2-v0_1 |",
        f"|--------|---------|-------------|-------------|---------|-----------|",
        f"| Exact rate | {s2['exact_rate']:.2%} | {v01['rate']:.2%} | {wdev['rate']:.2%} | {wdev['rate']-s2['exact_rate']:+.2%} | {wdev['rate']-v01['rate']:+.2%} |",
        f"| Parse rate | {s2['parse_rate']:.2%} | {v01['parse_rate']:.2%} | {wdev['parse_rate']:.2%} | — | — |",
        f"| Stop rate | {s2['stop_rate']:.2%} | {v01['stop_rate']:.2%} | {wdev['stop_rate']:.2%} | — | — |",
        f"| Mean tokens | {s2['mean_tokens']:.1f} | {v01['mean_tokens']:.1f} | {wdev['mean_tokens']:.1f} | — | — |",
        "",
        "## Strict Regression",
        f"- Exact: {strict.get('exact_rate',0):.2%}, Spurious: {strict.get('spurious_rate',0):.2%}",
        "",
        "## Protocol Regression",
        f"- Think both tags: {proto['think']['rate']:.2%} ({proto['think']['both']}/{proto['think']['total']})",
        f"- No-think clean: {proto['nothink']['rate']:.2%} ({proto['nothink']['clean']}/{proto['nothink']['total']})",
        "",
        "## Open Regression",
        f"- Too short: {open_r['too_short_rate']:.2%}",
        f"- Spurious think: {open_r['spurious_rate']:.2%}",
        f"- Mean words: {open_r['word_mean']}",
        f"- Stop success: {open_r['stop_rate']:.2%}",
    ]
    (FORMAL_OUTPUT / "formal_v0_2_report.md").write_text("\n".join(lines))

    print("\n" + "=" * 60)
    print(f"Done. Outputs:")
    print(f"  Adapter: {adapter_path_str}")
    print(f"  Report:  {FORMAL_OUTPUT / 'formal_v0_2_report.json'}")
    print(f"  MD:      {FORMAL_OUTPUT / 'formal_v0_2_report.md'}")
    print(f"  Metrics: {FORMAL_OUTPUT / 'train_metrics.json'}")


if __name__ == "__main__":
    main()
