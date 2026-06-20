#!/usr/bin/env python3
"""Stage 3 v0_3 data generator — all-examples median CoT with loop-safe cipher.

Fixes from v0_2:
  1. Gravity: ALL g values → median → query (not single-example)
  2. Unit: ALL ratios → median → query (not single-example)
  3. Cipher: ≤10 mapping entries, no fuzzy repeats, explicit decrypt line
  4. Bit: rule vector + query application
  5. Token budgets enforced per task type
  6. Deviations from spec recorded as skipped reasons
"""

from __future__ import annotations

import argparse, json, re, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_stage3_wonderland_cold_start import (
    TASK_BIT, TASK_NUMERAL, TASK_UNIT, TASK_GRAVITY, TASK_CIPHER, TASK_SYMBOLIC,
    THINK_OPEN, THINK_CLOSE, MAX_TOKEN_LENGTH,
    SAMPLE_ANSWER_ONLY, SAMPLE_COMPRESSED_COT,
    SAMPLE_STAGE1_5_REPLAY, SAMPLE_STAGE2_REPLAY,
    WONDERLAND_TASK_TYPES, REPLAY_SAMPLE_TYPES,
    ReasonerResult, ReasonerRegistry, QwenTokenCounter, Problem, Example,
    build_reasoner_registry, parse_wonderland_problem, task_type_for_problem,
    make_messages, build_thinking_completion,
    parse_thinking, parse_final_answer, final_answer_parse_ok,
    _base_wonderland_record, make_answer_only_record,
    build_answer_only_prompt, build_cot_prompt,
    validate_stage3_record,
    _read_train_csv_pool, read_stage3_sft_pool, select_wonderland_problems,
    _partition_train_dev,
    load_replay_rows, replay_candidate_ok, build_replay_rows_for_split,
    make_replay_record,
    build_report, build_audit_md, build_manual_review_md,
    write_json, write_text_atomic, write_jsonl_atomic, read_jsonl,
    prompt_hash, normalize_whitespace, get_user_and_assistant,
    _append_if_valid,
    _parse_float, _coefficient, _numeric_close, _format_compact_number,
    _extract_reasoner_answer, _extract_selected_rules, _clean_trace_line,
    _extract_symbolic_rule,
    EQUATION_NUMERIC_RE, BOXED_RE,
)

# ── Token budgets per task type (hard limits for compressed_cot) ──
TOKEN_BUDGET_HARD = {
    TASK_NUMERAL: 240,
    TASK_GRAVITY: 310,
    TASK_UNIT: 300,
    TASK_BIT: 400,
    TASK_CIPHER: 280,
    TASK_SYMBOLIC: 260,
}


# ═══════════════════════════════════════════════════════════════════════
# v0_3 trace builders
# ═══════════════════════════════════════════════════════════════════════

def _fmt_vals(values, max_digits=4):
    """Format a list of floats compactly."""
    if len(values) <= 5:
        return ", ".join(f"{v:.{max_digits}g}" for v in values)
    return ", ".join(f"{v:.{max_digits}g}" for v in values[:3]) + f", ..., {values[-1]:.{max_digits}g}"


def build_gravity_trace_v0_3(problem, all_g_values, median_g, query_t, predicted):
    """Gravity: ALL g values → median → query substitution."""
    median_text = _format_compact_number(median_g)
    g_list = _fmt_vals(sorted(all_g_values))
    return (
        f"Task type: gravity.\n"
        f"Compute g = 2*d/t^2 from each example.\n"
        f"The g values are: {g_list}.\n"
        f"Use the median g = {median_text}.\n"
        f"For query t = {query_t}, compute d = 0.5 * {median_text} * {query_t}^2 = {predicted:.2f}."
    )


def build_unit_trace_v0_3(problem, all_ratios, median_coef, query_value, predicted):
    """Unit: ALL ratios → median → query substitution."""
    coef_text = _format_compact_number(median_coef)
    ratio_list = _fmt_vals(sorted(all_ratios), max_digits=5)
    return (
        f"Task type: unit conversion.\n"
        f"Compute coefficient = output/input from each example.\n"
        f"The coefficients are: {ratio_list}.\n"
        f"Use the median coefficient = {coef_text}.\n"
        f"For query {query_value}, compute {query_value} * {coef_text} = {predicted:.2f}."
    )


def build_bit_trace_v0_3(problem, answer, rules, query):
    """Bit: rule vector + query application."""
    if not rules:
        rule_text = "rule vector inferred from examples"
    else:
        rule_text = ", ".join(f"b{i}:{r}" for i, r in enumerate(rules))
    return (
        f"Task type: bit manipulation.\n"
        f"Selected rules: {rule_text}.\n"
        f"For query {query}, applying these 8 rules gives {answer}."
    )


def build_cipher_trace_v0_3(problem, answer, raw_trace):
    """Cipher: ≤10 mapping entries, explicit decrypt line, no loops."""
    mappings = _extract_cipher_mapping_v0_3(raw_trace)
    mapping_text = ", ".join(mappings[:10])
    if len(mappings) > 10:
        mapping_text += " (remaining consistent)"
    return (
        f"Task type: cipher.\n"
        f"Key mappings needed for query: {mapping_text}.\n"
        f"Decrypt query \"{problem.question}\" -> \"{answer}\"."
    )


def _extract_cipher_mapping_v0_3(raw_trace):
    """Extract unique character mappings from cipher raw trace, deduplicated."""
    seen_pairs = set()
    result = []
    for line in raw_trace.splitlines():
        for m in re.findall(r"(\w)\s*[-:>→]+\s*(\w)", line):
            pair = (m[0], m[1])
            if pair not in seen_pairs and m[0] != m[1]:
                seen_pairs.add(pair)
                result.append(f"{m[0]}→{m[1]}")
    # Limit to 15 unique pairs (we only show 10 in trace)
    return result[:15]


def build_numeral_trace_v0_3(problem, answer, raw_trace):
    """Numeral: step-by-step greedy decomposition (same as v0_2)."""
    rule_text = _extract_numeral_steps(raw_trace)
    return (
        f"Task type: numeral conversion.\n"
        f"Rule: {rule_text}.\n"
        f"Convert {problem.question} → {answer}."
    )


def _extract_numeral_steps(raw_trace):
    """Extract greedy decomposition steps from numeral raw trace."""
    lines = raw_trace.splitlines()
    steps = []
    in_cv = False
    for line in lines:
        s = line.strip()
        if s.startswith("Converting"):
            in_cv = True
            steps.append(s.split(":")[0])
            continue
        if in_cv and s and (">=" in s or "->" in s):
            short = re.sub(r"remainder\s+", "rem ", s)
            steps.append(short)
        elif in_cv and s.startswith("Result:"):
            steps.append(s)
            in_cv = False
            break
    if steps:
        compressed = " | ".join(steps[:6])
        if len(steps) > 6:
            compressed += " | ..."
        return compressed
    return "Roman place-value rules from examples"


def build_symbolic_trace_v0_3(problem, answer, raw_trace):
    """Symbolic: specific rule + query → result. Returns '' if no specific rule."""
    rule_text = _extract_symbolic_rule(raw_trace)
    if not rule_text:
        return ""
    return (
        f"Task type: symbolic/equation transformation.\n"
        f"Rule: {rule_text}.\n"
        f"Query: {problem.question} → Result: {answer}."
    )


# ═══════════════════════════════════════════════════════════════════════
# v0_3 reasoners (gravity/unit return ALL intermediate values)
# ═══════════════════════════════════════════════════════════════════════

def reason_gravity_v0_3(problem):
    """Gravity reasoner returning all g values for median-based trace."""
    g_values = []
    raw_lines = ["gravity v0_3 median reasoner"]
    try:
        for ex in problem.examples:
            t = _parse_float(ex.input_value)
            d = _parse_float(ex.output_value)
            if t == 0: continue
            g = 2 * d / (t * t)
            g_values.append(g)
            raw_lines.append(f"g[{len(g_values)}] t={ex.input_value} d={ex.output_value} = {g:.12g}")
        if not g_values:
            return ReasonerResult(TASK_GRAVITY, "", "", "\n".join(raw_lines), False, "no_nonzero_examples", "low")
        median_g = float(statistics.median(g_values))
        query_t = _parse_float(problem.question)
        predicted = 0.5 * median_g * query_t * query_t
        gold = _parse_float(problem.answer)
    except Exception as exc:
        return ReasonerResult(TASK_GRAVITY, "", "", "\n".join(raw_lines), False, f"parse_error:{exc}", "low")
    raw_lines.extend([f"median_g={median_g:.12g}", f"query_t={problem.question}", f"predicted={predicted:.12g}", f"gold={problem.answer}"])
    if not _numeric_close(predicted, gold):
        raw_lines.append(f"tolerance_failed abs_err={abs(predicted-gold):.12g}")
        return ReasonerResult(TASK_GRAVITY, "", "", "\n".join(raw_lines), False, "numeric_tolerance_failed", "low")
    compressed = build_gravity_trace_v0_3(problem, [float(g) for g in g_values], median_g, _format_compact_number(query_t), predicted)
    return ReasonerResult(TASK_GRAVITY, problem.answer, compressed, "\n".join(raw_lines), True, "", "high",
                          {"all_g": [float(g) for g in g_values], "median_g": median_g, "predicted": predicted})


def reason_unit_v0_3(problem):
    """Unit reasoner returning all ratios for median-based trace."""
    ratios = []
    raw_lines = ["unit v0_3 median reasoner"]
    try:
        for ex in problem.examples:
            inp = _parse_float(ex.input_value)
            out = _parse_float(ex.output_value)
            if inp == 0: continue
            r = out / inp
            ratios.append(r)
            raw_lines.append(f"ratio[{len(ratios)}] {ex.input_value}->{ex.output_value} = {r:.12g}")
        if not ratios:
            return ReasonerResult(TASK_UNIT, "", "", "\n".join(raw_lines), False, "no_nonzero_examples", "low")
        median_coef = float(statistics.median(ratios))
        query_value = _parse_float(problem.question)
        predicted = query_value * median_coef
        gold = _parse_float(problem.answer)
    except Exception as exc:
        return ReasonerResult(TASK_UNIT, "", "", "\n".join(raw_lines), False, f"parse_error:{exc}", "low")
    raw_lines.extend([f"median_coef={median_coef:.12g}", f"query_value={problem.question}", f"predicted={predicted:.12g}", f"gold={problem.answer}"])
    if not _numeric_close(predicted, gold):
        raw_lines.append(f"tolerance_failed abs_err={abs(predicted-gold):.12g}")
        return ReasonerResult(TASK_UNIT, "", "", "\n".join(raw_lines), False, "numeric_tolerance_failed", "low")
    compressed = build_unit_trace_v0_3(problem, [float(r) for r in ratios], median_coef, problem.question, predicted)
    return ReasonerResult(TASK_UNIT, problem.answer, compressed, "\n".join(raw_lines), True, "", "high",
                          {"all_ratios": [float(r) for r in ratios], "median_coef": median_coef, "predicted": predicted})


def _safe_reason_legacy(problem, task_type, legacy_fn, builder_fn= None):
    """Run a legacy reasoner, build enriched trace, handle errors."""
    try:
        raw_trace = legacy_fn(problem)
    except Exception as exc:
        return ReasonerResult(task_type, "", "", "", False, f"reasoner_error:{exc}", "low")
    if not raw_trace:
        return ReasonerResult(task_type, "", "", "", False, "reasoner_returned_none", "low")
    answer = _extract_reasoner_answer(raw_trace).strip()
    if not answer:
        return ReasonerResult(task_type, "", "", raw_trace, False, "missing_reasoner_answer", "low")
    compressed = builder_fn(problem, answer, raw_trace) if builder_fn else ""
    if builder_fn and not compressed:
        return ReasonerResult(task_type, answer, "", raw_trace, False, "no_specific_rule_for_cot", "low")
    if builder_fn and "\\boxed" in compressed:
        return ReasonerResult(task_type, answer, "", raw_trace, False, "compressed_trace_boxed_residue", "low")
    return ReasonerResult(task_type, answer, compressed, raw_trace, True, "", "high")


def reason_bit_v0_3(problem):
    from stage3_reasoners.bit_manipulation import reasoning_bit_manipulation
    result = _safe_reason_legacy(problem, TASK_BIT, reasoning_bit_manipulation)
    if not result.ok:
        return result
    rules = _extract_selected_rules(result.raw_trace)
    compressed = build_bit_trace_v0_3(problem, result.answer, rules, problem.question)
    if "\\boxed" in compressed:
        return ReasonerResult(TASK_BIT, result.answer, "", result.raw_trace, False, "compressed_trace_boxed_residue", "low")
    return ReasonerResult(TASK_BIT, result.answer, compressed, result.raw_trace, True, "", "high")


def reason_cipher_v0_3(problem):
    from stage3_reasoners.cipher import reasoning_cipher
    result = _safe_reason_legacy(problem, TASK_CIPHER, reasoning_cipher)
    if not result.ok:
        return result
    compressed = build_cipher_trace_v0_3(problem, result.answer, result.raw_trace)
    if "\\boxed" in compressed:
        return ReasonerResult(TASK_CIPHER, result.answer, "", result.raw_trace, False, "compressed_trace_boxed_residue", "low")
    return ReasonerResult(TASK_CIPHER, result.answer, compressed, result.raw_trace, True, "", "high")


def reason_numeral_v0_3(problem):
    from stage3_reasoners.numeral import reasoning_numeral
    result = _safe_reason_legacy(problem, TASK_NUMERAL, reasoning_numeral)
    if not result.ok:
        return result
    compressed = build_numeral_trace_v0_3(problem, result.answer, result.raw_trace)
    if "\\boxed" in compressed:
        return ReasonerResult(TASK_NUMERAL, result.answer, "", result.raw_trace, False, "compressed_trace_boxed_residue", "low")
    return ReasonerResult(TASK_NUMERAL, result.answer, compressed, result.raw_trace, True, "", "high")


def reason_symbolic_v0_3(problem):
    from stage3_reasoners.cryptarithm import reasoning_cryptarithm
    from stage3_reasoners.equation_numeric import reasoning_equation_numeric
    fns = (reasoning_equation_numeric, reasoning_cryptarithm) if EQUATION_NUMERIC_RE.fullmatch(problem.question) else (reasoning_cryptarithm, reasoning_equation_numeric)
    errors = []
    for fn in fns:
        result = _safe_reason_legacy(problem, TASK_SYMBOLIC, fn, build_symbolic_trace_v0_3)
        if result.ok and result.compressed_trace:
            return result
        errors.append(result.error)
    return ReasonerResult(TASK_SYMBOLIC, "", "", "", False, f"symbolic_failed:{';'.join(errors)}", "low")


def build_v0_3_registry():
    return ReasonerRegistry({
        TASK_BIT: reason_bit_v0_3,
        TASK_NUMERAL: reason_numeral_v0_3,
        TASK_UNIT: reason_unit_v0_3,
        TASK_GRAVITY: reason_gravity_v0_3,
        TASK_CIPHER: reason_cipher_v0_3,
        TASK_SYMBOLIC: reason_symbolic_v0_3,
    })


# ═══════════════════════════════════════════════════════════════════════
# Token budget enforcement
# ═══════════════════════════════════════════════════════════════════════

def _cot_too_long(task_type, cot_record, token_counter):
    """Check if a compressed_cot record exceeds its token budget."""
    budget = TOKEN_BUDGET_HARD.get(task_type, 300)
    tokens = token_counter.count_messages(cot_record["messages"])
    return tokens > budget


def make_cot_record_v0_3(problem, result, index, token_counter, split):
    """Make CoT record with budget check."""
    record = _base_wonderland_record(
        record_id=f"stage3-{split}-wonderland-compressed_cot-{index:06d}-{problem.id}",
        problem=problem,
        sample_type=SAMPLE_COMPRESSED_COT,
        user=build_cot_prompt(problem),
        assistant=build_thinking_completion(result.compressed_trace, result.answer),
        token_counter=token_counter,
        reasoner_result=result,
    )
    if result.metadata:
        record["metadata"]["intermediate_values"] = dict(result.metadata)
    return record


# ═══════════════════════════════════════════════════════════════════════
# Wonderland row builder with budget enforcement
# ═══════════════════════════════════════════════════════════════════════

def build_wonderland_rows_v0_3(problems, *, split, registry, token_counter, skipped, task_skipped, raw_traces, reasoner_stats):
    rows = []
    seen_hashes = set()
    for index, problem in enumerate(problems):
        task_type = task_type_for_problem(problem)
        # answer_only always generated
        ao_row = make_answer_only_record(problem, index, token_counter)
        ao_row["id"] = f"stage3-{split}-wonderland-answer_only-{index:06d}-{problem.id}"
        _append_if_valid(rows, ao_row, skipped=skipped, seen_prompt_hashes=seen_hashes)

        reasoner_stats[task_type]["attempted"] += 1
        result = registry.run(problem)
        raw_traces.append({
            "source_id": problem.id, "split": split, "task_type": task_type,
            "ok": result.ok, "error": result.error, "answer": result.answer,
            "gold_answer": problem.answer, "raw_trace": result.raw_trace,
            "intermediate_values": result.metadata if result.metadata else {},
        })

        # Skip if reasoner failed
        if not result.ok:
            reason = f"cot_{task_type}_{result.error or 'reasoner_failed'}"
            skipped[reason] += 1
            task_skipped[task_type][reason] += 1
            continue

        # Skip if answer mismatch with gold
        if result.answer.strip() != problem.answer.strip():
            skipped["cot_answer_mismatch"] += 1
            task_skipped[task_type]["cot_answer_mismatch"] += 1
            continue

        reasoner_stats[task_type]["ok"] += 1
        cot_row = make_cot_record_v0_3(problem, result, index, token_counter, split)
        cot_row["id"] = f"stage3-{split}-wonderland-compressed_cot-{index:06d}-{problem.id}"

        # Token budget check
        if _cot_too_long(task_type, cot_row, token_counter):
            skipped[f"cot_{task_type}_over_token_budget"] += 1
            task_skipped[task_type]["cot_over_token_budget"] += 1
            continue

        # Validate
        before = len(rows)
        _append_if_valid(rows, cot_row, skipped=skipped, seen_prompt_hashes=seen_hashes)
        if len(rows) == before:
            task_skipped[task_type]["cot_validation_or_duplicate_rejected"] += 1
    return rows


# ═══════════════════════════════════════════════════════════════════════
# Ratio adjustment
# ═══════════════════════════════════════════════════════════════════════

def adjust_sample_ratios(rows, target_ao_ratio=0.32, seed=42):
    """Downsample answer_only to ~target_ao_ratio, keeping CoT + replays intact."""
    import random
    rng = random.Random(seed)
    answer_only = [r for r in rows if r["sample_type"] == SAMPLE_ANSWER_ONLY]
    compressed_cot = [r for r in rows if r["sample_type"] == SAMPLE_COMPRESSED_COT]
    replays = [r for r in rows if r["sample_type"] in REPLAY_SAMPLE_TYPES]

    n_cot = len(compressed_cot)
    n_replays = len(replays)
    # target: ao / (ao + cot + replays) = target_ao_ratio
    # ao = target_ao_ratio * (ao + cot + replays)
    # ao * (1 - target_ao_ratio) = target_ao_ratio * (cot + replays)
    # ao = target_ao_ratio / (1 - target_ao_ratio) * (cot + replays)
    target_ao = int(target_ao_ratio / (1 - target_ao_ratio) * (n_cot + n_replays))
    target_ao = max(0, min(target_ao, len(answer_only)))
    if target_ao < len(answer_only):
        rng.shuffle(answer_only)
        answer_only = answer_only[:target_ao]

    result = answer_only + compressed_cot + replays
    rng.shuffle(result)
    return result


# ═══════════════════════════════════════════════════════════════════════
# Trace quality report
# ═══════════════════════════════════════════════════════════════════════

def _write_trace_quality_report(rows_by_split, output_dir):
    import random
    rng = random.Random(42)
    train_rows = rows_by_split["train"]
    cot_rows = [r for r in train_rows if r["sample_type"] == SAMPLE_COMPRESSED_COT]
    by_tt = defaultdict(list)
    for r in cot_rows:
        by_tt[r["task_type"]].append(r)

    lines = [
        "# Stage 3 v0_3 Compressed CoT Trace Quality Report",
        "",
        f"Total compressed_cot in train: {len(cot_rows)}",
        "",
        "| Task | Sampled | Numeric Intermediate | Rule Name | Query Sub | Final Answer |",
        "|------|---------|---------------------|-----------|-----------|-------------|",
    ]
    for tt in sorted(by_tt.keys()):
        recs = by_tt[tt]
        sample = rng.sample(recs, min(20, len(recs)))
        ni = sum(1 for r in sample if _has_numeric_intermediate(r["messages"][1]["content"], tt))
        rn = sum(1 for r in sample if _has_rule_name(r["messages"][1]["content"], tt))
        qs = sum(1 for r in sample if _has_query_sub(r["messages"][1]["content"], tt))
        fa = sum(1 for r in sample if _has_final_answer(r["messages"][1]["content"]))
        lines.append(f"| {tt} | {len(sample)} | {ni}/{len(sample)} ({ni/len(sample):.0%}) | {rn}/{len(sample)} | {qs}/{len(sample)} | {fa}/{len(sample)} |")
    lines.append("")
    lines.append("## Sample traces")
    for tt in sorted(by_tt.keys()):
        recs = by_tt[tt]
        sample = rng.sample(recs, min(3, len(recs)))
        lines.append(f"\n### {tt}")
        for i, r in enumerate(sample):
            assistant = r["messages"][1]["content"]
            lines.append(f"\n**Sample {i+1}** (`{r['id']}`):")
            lines.append("```")
            lines.append(assistant[:600])
            lines.append("```")
    (output_dir / "stage3_trace_quality_report.md").write_text("\n".join(lines))


def _has_numeric_intermediate(assistant, tt):
    if tt in (TASK_GRAVITY, TASK_UNIT):
        return bool(re.search(r"=\s*\d+\.?\d*", assistant))
    if tt == TASK_BIT:
        return bool(re.search(r"b\d:", assistant))
    if tt == TASK_CIPHER:
        return bool(re.search(r"\w→\w", assistant))
    return False


def _has_rule_name(assistant, tt):
    return bool(re.search(r"Rule:|Mapping:|Selected rules:|median g|median coefficient", assistant, re.IGNORECASE))


def _has_query_sub(assistant, tt):
    return bool(re.search(r"query|convert|decrypt|apply|For query", assistant, re.IGNORECASE))


def _has_final_answer(assistant):
    return bool(assistant.split("</think>")[-1].strip() if "</think>" in assistant else False)


# ═══════════════════════════════════════════════════════════════════════
# Splits validation
# ═══════════════════════════════════════════════════════════════════════

def validate_all_splits(rows_by_split):
    ids = set()
    for split, rows in rows_by_split.items():
        split_hashes = set()
        for row in rows:
            rid = str(row["id"])
            if rid in ids:
                raise ValueError(f"duplicate id: {rid}")
            ids.add(rid)
            validate_stage3_record(row)
            h = str(row["metadata"]["prompt_hash"])
            if h in split_hashes:
                raise ValueError(f"duplicate prompt hash in {split}: {h[:12]}")
            split_hashes.add(h)


# ═══════════════════════════════════════════════════════════════════════
# Extended manual review (gravity 10, unit 10, cipher 10, etc.)
# ═══════════════════════════════════════════════════════════════════════

def build_manual_review_md_v0_3(rows_by_split, raw_traces, report):
    import random
    rng = random.Random(42)
    lines = [
        "# Stage 3 v0_3 Manual Review",
        "",
        f"## Summary",
        f"- sample_total: {report['sample_total']}",
        f"- splits: {json.dumps(report['splits'], ensure_ascii=False)}",
        "",
    ]

    def append_row(row, split, number):
        user, assistant = get_user_and_assistant(row["messages"])
        meta = row["metadata"]
        lines.extend([
            f"### {number}. {split} / {row['sample_type']} / {row['task_type']}",
            f"- id: `{row['id']}`",
            f"- token_length: {meta.get('token_length')}",
            f"- gold_answer: {meta.get('gold_answer')}",
            f"- final_answer: {meta.get('final_answer')}",
        ])
        if meta.get('intermediate_values'):
            lines.append(f"- intermediates: {json.dumps(meta['intermediate_values'], ensure_ascii=False)}")
        if meta.get('reasoner') and isinstance(meta['reasoner'], dict):
            lines.append(f"- reasoner: ok={meta['reasoner'].get('ok')} error={meta['reasoner'].get('error','')} confidence={meta['reasoner'].get('confidence','')}")
        lines.extend([
            "",
            "User:",
            "```text",
            user[:800],
            "```",
            "",
            "Assistant:",
            "```text",
            assistant[:1200],
            "```",
            "",
        ])

    # Group rows by type for sampling
    all_rows = [r for rows in rows_by_split.values() for r in rows]
    by_type = defaultdict(list)
    for r in all_rows:
        key = f"{r['task_type']}/{r['sample_type']}"
        by_type[key].append(r)

    # Sampling targets
    targets = {
        f"{TASK_GRAVITY}/{SAMPLE_COMPRESSED_COT}": 10,
        f"{TASK_UNIT}/{SAMPLE_COMPRESSED_COT}": 10,
        f"{TASK_CIPHER}/{SAMPLE_COMPRESSED_COT}": 10,
        f"{TASK_NUMERAL}/{SAMPLE_COMPRESSED_COT}": 5,
        f"{TASK_BIT}/{SAMPLE_COMPRESSED_COT}": 5,
        f"{TASK_SYMBOLIC}/{SAMPLE_COMPRESSED_COT}": 5,
        f"{TASK_GRAVITY}/{SAMPLE_ANSWER_ONLY}": 2,
        f"{TASK_UNIT}/{SAMPLE_ANSWER_ONLY}": 2,
        f"{TASK_CIPHER}/{SAMPLE_ANSWER_ONLY}": 2,
        f"{TASK_NUMERAL}/{SAMPLE_ANSWER_ONLY}": 2,
        f"{SAMPLE_STAGE1_5_REPLAY}/{SAMPLE_STAGE1_5_REPLAY}": 5,
        f"{SAMPLE_STAGE2_REPLAY}/{SAMPLE_STAGE2_REPLAY}": 5,
    }

    shown = 0
    for key, count in sorted(targets.items()):
        pool = by_type.get(key, [])
        sample = rng.sample(pool, min(count, len(pool)))
        tt_label = key.split("/")[0]
        st_label = key.split("/")[1]
        lines.append(f"\n## {tt_label} / {st_label} ({len(pool)} total, {len(sample)} shown)")
        for i, row in enumerate(sample):
            shown += 1
            split = "train" if row["id"].startswith("stage3-train") else "dev"
            append_row(row, split, shown)

    return "\n".join(lines) + "\n"


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def generate_v0_3(*, input_csv, split_path, output_dir, seed=42, dev_ratio=0.05,
                   max_stage3_prompts=0, stage1_5_replay_path=None,
                   stage2_replay_path=None, replay_train_limit=200,
                   replay_dev_limit=40, dry_run=False):
    import random
    token_counter = QwenTokenCounter(model_id="models/Qwen3-1.7B-Base", cache_dir=Path(".hf-cache/hub"))
    skipped = Counter()
    task_skipped = defaultdict(Counter)
    reasoner_stats = defaultdict(Counter)
    raw_traces = []
    registry = build_v0_3_registry()

    problems, pool_ids, missing_ids = select_wonderland_problems(
        input_csv=input_csv, split_path=split_path,
        max_stage3_prompts=max_stage3_prompts, seed=seed, skipped=skipped,
    )
    train_problems, dev_problems = _partition_train_dev(problems, dev_ratio)

    stage1_5_rows = load_replay_rows(stage1_5_replay_path, sample_type=SAMPLE_STAGE1_5_REPLAY, skipped=skipped)
    stage2_rows = load_replay_rows(stage2_replay_path, sample_type=SAMPLE_STAGE2_REPLAY, skipped=skipped)

    rows_by_split = {"train": [], "dev": []}

    for split_name, probs in [("train", train_problems), ("dev", dev_problems)]:
        rows_by_split[split_name].extend(
            build_wonderland_rows_v0_3(
                probs, split=split_name, registry=registry,
                token_counter=token_counter, skipped=skipped,
                task_skipped=task_skipped, raw_traces=raw_traces,
                reasoner_stats=reasoner_stats,
            )
        )

    for split, limit in (("train", replay_train_limit), ("dev", replay_dev_limit)):
        if limit <= 0: continue
        rows_by_split[split].extend(
            build_replay_rows_for_split(stage1_5_rows, split=split, sample_type=SAMPLE_STAGE1_5_REPLAY,
                                        limit=limit, seed=seed+(11 if split=="train" else 12),
                                        token_counter=token_counter, skipped=skipped))
        rows_by_split[split].extend(
            build_replay_rows_for_split(stage2_rows, split=split, sample_type=SAMPLE_STAGE2_REPLAY,
                                        limit=limit, seed=seed+(21 if split=="train" else 22),
                                        token_counter=token_counter, skipped=skipped))

    # Deduplicate
    for sn in rows_by_split:
        seen = set(); deduped = []
        for r in rows_by_split[sn]:
            h = str(r["metadata"]["prompt_hash"])
            if h in seen: skipped["duplicate_prompt_hash_across_replay"] += 1; continue
            seen.add(h); deduped.append(r)
        rows_by_split[sn] = deduped

    # Ratio adjustment
    pre = Counter(r["sample_type"] for r in rows_by_split["train"])
    print(f"[v0_3] Pre-adjust train: {dict(pre)}")
    rows_by_split["train"] = adjust_sample_ratios(rows_by_split["train"], target_ao_ratio=0.32, seed=seed+99)
    post = Counter(r["sample_type"] for r in rows_by_split["train"])
    print(f"[v0_3] Post-adjust train: {dict(post)}")

    validate_all_splits(rows_by_split)

    report = build_report(rows_by_split, skipped=skipped, task_skipped=task_skipped,
                          reasoner_stats=reasoner_stats, pool_ids=pool_ids,
                          missing_split_ids=missing_ids, tokenizer=token_counter)

    for sn in rows_by_split:
        counts = Counter(r["sample_type"] for r in rows_by_split[sn])
        total = len(rows_by_split[sn])
        print(f"[v0_3] {sn}: total={total}, ratios: {{{', '.join(f'{k}: {v/total:.1%}' for k,v in counts.most_common())}}}")

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl_atomic(output_dir / "train.jsonl", rows_by_split["train"])
        write_jsonl_atomic(output_dir / "dev.jsonl", rows_by_split["dev"])
        write_json(output_dir / "report.json", report)
        write_text_atomic(output_dir / "audit.md", build_audit_md(report))
        manual = build_manual_review_md_v0_3(rows_by_split, raw_traces, report)
        write_text_atomic(output_dir / "manual_review.md", manual)
        write_jsonl_atomic(output_dir / "debug" / "raw_traces.jsonl", raw_traces)
        _write_trace_quality_report(rows_by_split, output_dir)
        # Symlink validation → dev
        symlink = output_dir / "validation.jsonl"
        if not symlink.exists():
            symlink.symlink_to("dev.jsonl")

    return rows_by_split, report


def main():
    p = argparse.ArgumentParser(description="Generate Stage 3 v0_3 data with median-based CoT.")
    p.add_argument("--input-csv", type=Path, default=Path("data/raw/train.csv"))
    p.add_argument("--split-path", type=Path, default=Path("splits/wonderland_split_seed42.json"))
    p.add_argument("--output-dir", type=Path, default=Path("data/instruction/stage3_wonderland_cold_start_v0_3"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dev-ratio", type=float, default=0.05)
    p.add_argument("--max-stage3-prompts", type=int, default=0)
    p.add_argument("--stage1-5-replay-path", type=Path, default=Path("data/instruction/stage1_5/train.jsonl"))
    p.add_argument("--stage2-replay-path", type=Path, default=Path("data/instruction/stage2_thinking/train.jsonl"))
    p.add_argument("--replay-train-limit", type=int, default=200)
    p.add_argument("--replay-dev-limit", type=int, default=40)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    generate_v0_3(
        input_csv=args.input_csv, split_path=args.split_path, output_dir=args.output_dir,
        seed=args.seed, dev_ratio=args.dev_ratio, max_stage3_prompts=args.max_stage3_prompts,
        stage1_5_replay_path=args.stage1_5_replay_path, stage2_replay_path=args.stage2_replay_path,
        replay_train_limit=args.replay_train_limit, replay_dev_limit=args.replay_dev_limit,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
