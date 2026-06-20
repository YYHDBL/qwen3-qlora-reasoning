#!/usr/bin/env python3
"""Task 3: Audit compressed CoT teaching signal density.
Output: outputs/stage3_wonderland_coldstart/stage3_trace_quality_report.md
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TRAIN_PATH = Path("data/instruction/stage3_wonderland_cold_start/train.jsonl")
OUT_DIR = Path("outputs/stage3_wonderland_coldstart")
SEED = 42


def classify_trace(assistant_content, task_type):
    """Classify a compressed CoT trace by teaching signal density.
    Returns dict with:
      - only_template_trace: bool — trace is just task_type tag + generic steps, no specifics
      - has_rule_name: bool — mentions specific rule/pattern name
      - has_numeric_intermediate: bool — contains concrete numeric intermediate values
      - has_query_substitution: bool — shows query value substitution
      - has_final_answer: bool — final answer present
    """
    import re
    content = assistant_content

    result = {
        "only_template_trace": False,
        "has_rule_name": False,
        "has_numeric_intermediate": False,
        "has_query_substitution": False,
        "has_final_answer": False,
        "details": "",
    }

    # Extract think block
    think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    if not think_match:
        result["details"] = "no think block"
        return result

    think = think_match.group(1).strip()
    answer = content.split("</think>")[-1].strip()

    # has_final_answer
    result["has_final_answer"] = bool(answer)

    # Task-type-specific checks
    if task_type == "gravity":
        # Check for numeric g value
        result["has_numeric_intermediate"] = bool(re.search(r"g\s*[≈=]\s*\d+\.?\d*", think))
        result["has_query_substitution"] = bool(re.search(r"query|t\s*=\s*\d+|compute d", think, re.IGNORECASE))
        result["has_rule_name"] = bool(re.search(r"g\s*=\s*2\*d/\s*t\^2|0\.5\*g\*t\^2", think))
        if not result["has_numeric_intermediate"] and not result["has_query_substitution"]:
            result["only_template_trace"] = True
            result["details"] = "template only: no g value, no query substitution"

    elif task_type == "unit_conversion":
        result["has_numeric_intermediate"] = bool(re.search(r"ratio.*?\d+\.\d+|coefficient.*?\d+\.\d+|\d+\.\d+.*?ratio", think, re.IGNORECASE))
        result["has_query_substitution"] = bool(re.search(r"query|apply.*coefficient|multiply|×", think, re.IGNORECASE))
        result["has_rule_name"] = bool(re.search(r"ratio|output/input", think, re.IGNORECASE))
        if not result["has_numeric_intermediate"] and not result["has_query_substitution"]:
            result["only_template_trace"] = True
            result["details"] = "template only: no ratio value, no query substitution"

    elif task_type == "bit_manipulation":
        result["has_rule_name"] = bool(re.search(r"selected rule|rule:|pattern:", think, re.IGNORECASE))
        result["has_numeric_intermediate"] = bool(re.search(r"rule\s*[=:]\s*\[\d|bit\s*\d", think, re.IGNORECASE))
        result["has_query_substitution"] = bool(re.search(r"apply.*rule|query|each bit", think, re.IGNORECASE))
        if not result["has_rule_name"] and not result["has_numeric_intermediate"]:
            result["only_template_trace"] = True
            result["details"] = "template only: no rule vector"

    elif task_type == "cipher":
        result["has_rule_name"] = bool(re.search(r"mapping|substitution|shift|caesar", think, re.IGNORECASE))
        result["has_numeric_intermediate"] = bool(re.search(r"\w+\s*->\s*\w+|encrypted|decoded", think, re.IGNORECASE))
        result["has_query_substitution"] = bool(re.search(r"query|apply.*mapping|decode.*word", think, re.IGNORECASE))
        if not result["has_numeric_intermediate"]:
            result["only_template_trace"] = True
            result["details"] = "template only: no mapping/word decoding"

    elif task_type == "symbolic_equation":
        result["has_rule_name"] = bool(re.search(r"rule:|reversed|swap|mirror", think, re.IGNORECASE))
        result["has_query_substitution"] = bool(re.search(r"apply|query|each.*rule|operands", think, re.IGNORECASE))
        if not result["has_rule_name"]:
            result["only_template_trace"] = True
            result["details"] = "template only: no rule name"

    elif task_type == "numeral":
        result["has_rule_name"] = bool(re.search(r"rule|place.?value|roman|decimal", think, re.IGNORECASE))
        result["has_query_substitution"] = bool(re.search(r"convert|query|number", think, re.IGNORECASE))
        if not result["has_rule_name"]:
            result["only_template_trace"] = True
            result["details"] = "template only: no rule reference"

    return result


def audit_all():
    random.seed(SEED)
    records = [json.loads(l) for l in open(TRAIN_PATH) if l.strip()]

    # Filter compressed_cot only
    cot_records = [r for r in records if r["sample_type"] == "wonderland_compressed_cot"]
    by_tt = defaultdict(list)
    for r in cot_records:
        by_tt[r["task_type"]].append(r)

    print(f"Total compressed_cot records: {len(cot_records)}")
    print(f"Task types: {sorted(by_tt.keys())}")
    for tt, recs in sorted(by_tt.items()):
        print(f"  {tt}: {len(recs)}")

    # Sample 20 per task type (or all if fewer)
    report_lines = [
        "# Stage 3 Compressed CoT Trace Quality Report",
        "",
        "Audit of teaching signal density in compressed_cot traces for each task type.",
        "",
        f"Total compressed_cot in train: {len(cot_records)}",
        "",
    ]

    all_results = {}

    for tt in sorted(by_tt.keys()):
        recs = by_tt[tt]
        sample = random.sample(recs, min(20, len(recs)))
        results = []
        for r in sample:
            assistant = r["messages"][-1]["content"]
            c = classify_trace(assistant, tt)
            c["id"] = r["id"]
            results.append(c)

        all_results[tt] = results

        # Stats
        n = len(results)
        stats = {
            "total_sampled": n,
            "total_in_train": len(recs),
            "only_template_trace": sum(1 for r in results if r["only_template_trace"]),
            "has_rule_name": sum(1 for r in results if r["has_rule_name"]),
            "has_numeric_intermediate": sum(1 for r in results if r["has_numeric_intermediate"]),
            "has_query_substitution": sum(1 for r in results if r["has_query_substitution"]),
            "has_final_answer": sum(1 for r in results if r["has_final_answer"]),
        }

        report_lines.append(f"## {tt}")
        report_lines.append("")
        report_lines.append("| Metric | Count / Total | Rate |")
        report_lines.append("|--------|---------------|------|")
        for k, v in stats.items():
            if k in ("total_sampled", "total_in_train"):
                report_lines.append(f"| {k} | {v} | — |")
            else:
                report_lines.append(f"| {k} | {v}/{n} | {v/n:.1%} |")
        report_lines.append("")

        # Show problematic traces
        template_only = [r for r in results if r["only_template_trace"]]
        if template_only:
            report_lines.append(f"**Warning**: {len(template_only)}/{n} traces are template-only (no specific intermediates).")
            for r in template_only[:3]:
                report_lines.append(f"  - `{r['id']}`: {r['details']}")
            report_lines.append("")

        # Show sample trace
        report_lines.append("### Sample trace")
        report_lines.append("```")
        report_lines.append(results[0].get("details", ""))
        report_lines.append("```")
        report_lines.append("")

    # Summary
    report_lines.insert(4, "## Summary")
    report_lines.insert(5, "")
    for tt in sorted(all_results.keys()):
        r = all_results[tt]
        n = len(r)
        tp = sum(1 for x in r if x["only_template_trace"])
        ni = sum(1 for x in r if x["has_numeric_intermediate"])
        report_lines.insert(6, f"| {tt} | {n} | {tp}/{n} ({tp/n:.0%}) | {ni}/{n} ({ni/n:.0%}) | {sum(1 for x in r if x['has_rule_name'])/n:.0%} | {sum(1 for x in r if x['has_query_substitution'])/n:.0%} |")
    report_lines.insert(6, "| Task | Sampled | Template-only | Numeric Intermediate | Rule Name | Query Sub |")
    report_lines.insert(6, "|------|---------|---------------|---------------------|-----------|----------|")

    report = "\n".join(report_lines)
    (OUT_DIR / "stage3_trace_quality_report.md").write_text(report)

    print("\n=== Audit Summary ===")
    for tt in sorted(all_results.keys()):
        r = all_results[tt]
        n = len(r)
        tp = sum(1 for x in r if x["only_template_trace"])
        ni = sum(1 for x in r if x["has_numeric_intermediate"])
        rn = sum(1 for x in r if x["has_rule_name"])
        print(f"  {tt:<22}: template_only={tp}/{n}  numeric={ni}/{n}  rule_name={rn}/{n}")

    # Also dump JSON for reference
    json.dump(all_results, open(OUT_DIR / "trace_quality_audit.json", "w"), indent=2, default=str)
    print(f"\nReport written to {OUT_DIR / 'stage3_trace_quality_report.md'}")


if __name__ == "__main__":
    audit_all()
