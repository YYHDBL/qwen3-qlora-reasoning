#!/usr/bin/env python3
"""Stage 3 v0_2 data generator — enriched CoT with concrete numeric intermediates.

Key changes from v0_1:
  1. Gravity CoT: show g computation from one example + query substitution with t value
  2. Unit CoT: show ratio computation from one example + query * coefficient
  3. Bit CoT: show selected rule vector + query application
  4. Cipher CoT: show key mapping + word decoding
  5. Symbolic: if no specific rule, degrade to answer-only (not compressed_cot)
  6. Sample ratios: 30-35% answer_only, 45-50% compressed_cot, 10% strict, 10% thinking
  7. Store intermediate values in metadata for audit

Does NOT read Wonderland validation/test. Does NOT train. Does NOT touch adapters.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Import all core functions from the original generator
from generate_stage3_wonderland_cold_start import (
    # Constants
    TASK_BIT, TASK_NUMERAL, TASK_UNIT, TASK_GRAVITY, TASK_CIPHER, TASK_SYMBOLIC,
    THINK_OPEN, THINK_CLOSE, MAX_TOKEN_LENGTH,
    SAMPLE_ANSWER_ONLY, SAMPLE_COMPRESSED_COT,
    SAMPLE_STAGE1_5_REPLAY, SAMPLE_STAGE2_REPLAY,
    WONDERLAND_TASK_TYPES, REPLAY_SAMPLE_TYPES,
    # Classes
    ReasonerResult, ReasonerRegistry, QwenTokenCounter, Problem, Example,
    # Core functions
    build_reasoner_registry, parse_wonderland_problem, task_type_for_problem,
    make_messages, build_thinking_completion,
    parse_thinking, parse_final_answer, final_answer_parse_ok,
    _base_wonderland_record, make_answer_only_record,
    build_answer_only_prompt, build_cot_prompt,
    validate_stage3_record,
    # CSV/split functions
    _read_train_csv_pool, read_stage3_sft_pool, select_wonderland_problems,
    _partition_train_dev,
    load_replay_rows, replay_candidate_ok, build_replay_rows_for_split,
    make_replay_record,
    # Report
    build_report, build_audit_md, build_manual_review_md,
    # I/O
    write_json, write_text_atomic, write_jsonl_atomic, read_jsonl,
    prompt_hash, normalize_whitespace, get_user_and_assistant,
    _append_if_valid,
    # Reasoner internals needed for intermediate values
    _parse_float, _coefficient, _numeric_close, _format_compact_number,
    _extract_reasoner_answer, _extract_selected_rules, _clean_trace_line,
    _extract_symbolic_rule,
    # Reasoners
    reason_bit_manipulation, reason_numeral, reason_unit_conversion,
    reason_gravity, reason_cipher, reason_symbolic_equation,
    # Trace builders (we'll override these)
    build_bit_trace as _old_build_bit_trace,
    build_numeral_trace as _old_build_numeral_trace,
    build_unit_trace as _old_build_unit_trace,
    build_gravity_trace as _old_build_gravity_trace,
    build_cipher_trace as _old_build_cipher_trace,
    build_symbolic_trace as _old_build_symbolic_trace,
    # Other
    EQUATION_NUMERIC_RE, BOXED_RE,
    BIT_PREFIX, GRAVITY_PREFIX, UNIT_PREFIX, CIPHER_PREFIX, NUMERAL_PREFIX, SYMBOLIC_PREFIX,
)


# ═══════════════════════════════════════════════════════════════════════
# v0_2 enriched trace builders
# ═══════════════════════════════════════════════════════════════════════

def build_gravity_trace_v0_2(problem, g_value, query_t, predicted):
    """Gravity CoT with concrete g computation + query substitution."""
    g_text = _format_compact_number(g_value)
    # Pick first example to demonstrate g computation
    t1, d1 = problem.examples[0].input_value, problem.examples[0].output_value
    return (
        f"Task type: gravity.\n"
        f"Use g = 2*d/t^2. Example: for t={t1}, d={d1}, g = 2*{d1}/{t1}^2 ≈ {g_text}.\n"
        f"For query t={query_t}, compute d = 0.5 * {g_text} * {query_t}^2 = {predicted:.2f}. Round to required format."
    )


def build_unit_trace_v0_2(problem, coef, query_value, predicted):
    """Unit conversion CoT with concrete ratio computation + query application."""
    coef_text = _format_compact_number(coef)
    # Pick first example to demonstrate ratio
    inp1, out1 = problem.examples[0].input_value, problem.examples[0].output_value
    ratio1 = float(out1) / float(inp1) if float(inp1) != 0 else 0
    return (
        f"Task type: unit conversion.\n"
        f"Compute output/input ratios. Example: {out1}/{inp1} ≈ {_format_compact_number(ratio1)}.\n"
        f"Median coefficient = {coef_text}. For query {query_value}: {query_value} * {coef_text} = {predicted:.2f}. Round to required format."
    )


def build_bit_trace_v0_2(problem, answer, rules, query):
    """Bit CoT with selected rule vector + query application."""
    rule_parts = []
    for i, r in enumerate(rules):
        rule_parts.append(f"b{i}:{r}")
    rule_text = ", ".join(rule_parts) if rule_parts else "rule vector inferred from examples"
    return (
        f"Task type: bit_manipulation.\n"
        f"Selected rules: {rule_text}.\n"
        f"Apply bit-by-bit to query {query} → {answer}."
    )


def build_cipher_trace_v0_2(problem, answer, raw_trace):
    """Cipher CoT with key mapping extraction."""
    # Try to extract mapping from raw trace
    mapping_text = _extract_cipher_mapping(raw_trace, problem)
    return (
        f"Task type: cipher.\n"
        f"Infer substitution mapping from examples.\n"
        f"Mapping: {mapping_text}.\n"
        f"Decrypt query '{problem.question}' → '{answer}'."
    )


def _extract_cipher_mapping(raw_trace, problem):
    """Extract character mapping from cipher raw trace."""
    # Look for mapping lines in raw trace
    mappings = []
    for line in raw_trace.splitlines():
        # Pattern: char -> char or {char: char}
        m = re.findall(r"(\w)\s*[-:>→]+\s*(\w)", line)
        for a, b in m:
            if a != b:
                mappings.append(f"{a}→{b}")
    if mappings:
        seen = set()
        unique = []
        for m in mappings:
            if m not in seen:
                seen.add(m)
                unique.append(m)
        return ", ".join(unique[:10]) + (", ..." if len(unique) > 10 else "")
    return "inferred from input-output pairs"


def build_numeral_trace_v0_2(problem, answer, raw_trace):
    """Numeral CoT with conversion mapping."""
    # Try to extract mapping from raw trace
    rules_text = _extract_numeral_rules(raw_trace)
    return (
        f"Task type: numeral conversion.\n"
        f"Rule: {rules_text}.\n"
        f"Convert {problem.question} → {answer}."
    )


def _extract_numeral_rules(raw_trace):
    """Extract Roman numeral conversion steps from raw trace."""
    # Look for "Converting N:" line + step-by-step decomposition
    lines = raw_trace.splitlines()
    steps = []
    in_conversion = False
    for line in lines:
        s = line.strip()
        if s.startswith("Converting"):
            in_conversion = True
            steps.append(s.split(":")[0])
            continue
        if in_conversion and s and (">=" in s or "->" in s):
            # Capture: "61 >= 50 -> L, remainder 11"
            steps.append(s)
            continue
        if in_conversion and s.startswith("Result:"):
            steps.append(s)
            in_conversion = False
            continue
        if in_conversion and not s:
            in_conversion = False
    if steps:
        compressed = " | ".join(steps[:6])
        if len(steps) > 6:
            compressed += " | ..."
        return compressed
    # Fallback: look for mappings
    mappings = re.findall(r"(\d+)\s*>=?\s*(\d+)\s*->\s*(\w+)", raw_trace)
    if mappings:
        parts = [f"{num}→{sym}" for num, _, sym in mappings[:6]]
        return "Roman: " + ", ".join(parts)
    return "Roman place-value rules from examples"


def build_symbolic_trace_v0_2(problem, answer, raw_trace):
    """Symbolic CoT with rule name. Returns '' if no specific rule."""
    rule_text = _extract_symbolic_rule(raw_trace)
    if not rule_text:
        return ""
    # Try to show application
    app_text = f"Query: {problem.question} → Result: {answer}."
    return (
        f"Task type: symbolic/equation-like transformation.\n"
        f"Rule: {rule_text}.\n"
        f"{app_text}"
    )


# ═══════════════════════════════════════════════════════════════════════
# Override reasoner to also return intermediate values
# ═══════════════════════════════════════════════════════════════════════

def reason_gravity_v0_2(problem):
    """Gravity reasoner that returns intermediates for enriched trace."""
    g_values = []
    raw_lines = ["local deterministic gravity reasoner v0_2"]
    try:
        for ex in problem.examples:
            t = _parse_float(ex.input_value)
            distance = _parse_float(ex.output_value)
            if t == 0:
                continue
            g = 2 * distance / (t * t)
            g_values.append(g)
            raw_lines.append(f"example t={ex.input_value} distance={ex.output_value} g={g:.12g}")
        if not g_values:
            return ReasonerResult(TASK_GRAVITY, "", "", "\n".join(raw_lines), False, "no_nonzero_examples", "low")
        g_value = _coefficient(g_values)
        query_t = _parse_float(problem.question)
        predicted = 0.5 * g_value * query_t * query_t
        gold = _parse_float(problem.answer)
    except Exception as exc:
        return ReasonerResult(TASK_GRAVITY, "", "", "\n".join(raw_lines), False, f"parse_error:{exc}", "low")

    raw_lines.extend([
        f"g={g_value:.12g}", f"query_time={problem.question}",
        f"predicted={predicted:.12g}", f"gold={problem.answer}",
    ])
    if not _numeric_close(predicted, gold):
        raw_lines.append(f"numeric_tolerance_failed abs_error={abs(predicted - gold):.12g}")
        return ReasonerResult(TASK_GRAVITY, "", "", "\n".join(raw_lines), False, "numeric_tolerance_failed", "low")

    compressed = build_gravity_trace_v0_2(problem, g_value, _format_compact_number(query_t), predicted)
    return ReasonerResult(
        TASK_GRAVITY, problem.answer, compressed, "\n".join(raw_lines), True, "", "high",
        {"g": g_value, "predicted": predicted, "query_t": query_t}
    )


def reason_unit_v0_2(problem):
    """Unit reasoner that returns intermediates for enriched trace."""
    ratios = []
    raw_lines = ["local deterministic unit_conversion reasoner v0_2"]
    try:
        for ex in problem.examples:
            inp = _parse_float(ex.input_value)
            out = _parse_float(ex.output_value)
            if inp == 0:
                continue
            ratio = out / inp
            ratios.append(ratio)
            raw_lines.append(f"example input={ex.input_value} output={ex.output_value} ratio={ratio:.12g}")
        if not ratios:
            return ReasonerResult(TASK_UNIT, "", "", "\n".join(raw_lines), False, "no_nonzero_examples", "low")
        coef = _coefficient(ratios)
        query_value = _parse_float(problem.question)
        predicted = query_value * coef
        gold = _parse_float(problem.answer)
    except Exception as exc:
        return ReasonerResult(TASK_UNIT, "", "", "\n".join(raw_lines), False, f"parse_error:{exc}", "low")

    raw_lines.extend([
        f"coefficient={coef:.12g}", f"query_value={problem.question}",
        f"predicted={predicted:.12g}", f"gold={problem.answer}",
    ])
    if not _numeric_close(predicted, gold):
        raw_lines.append(f"numeric_tolerance_failed abs_error={abs(predicted - gold):.12g}")
        return ReasonerResult(TASK_UNIT, "", "", "\n".join(raw_lines), False, "numeric_tolerance_failed", "low")

    compressed = build_unit_trace_v0_2(problem, coef, problem.question, predicted)
    return ReasonerResult(
        TASK_UNIT, problem.answer, compressed, "\n".join(raw_lines), True, "", "high",
        {"coefficient": coef, "predicted": predicted}
    )


def reason_bit_v0_2(problem):
    """Bit reasoner with enriched trace (rule vector + application)."""
    from stage3_reasoners.bit_manipulation import reasoning_bit_manipulation

    try:
        raw_trace = reasoning_bit_manipulation(problem)
    except Exception as exc:
        return ReasonerResult(TASK_BIT, "", "", "", False, f"reasoner_error:{exc}", "low")
    if not raw_trace:
        return ReasonerResult(TASK_BIT, "", "", "", False, "reasoner_returned_none", "low")

    answer = _extract_reasoner_answer(raw_trace).strip()
    if not answer:
        return ReasonerResult(TASK_BIT, "", "", raw_trace, False, "missing_reasoner_answer", "low")

    rules = _extract_selected_rules(raw_trace)
    compressed = build_bit_trace_v0_2(problem, answer, rules, problem.question)
    if "\\boxed" in compressed:
        return ReasonerResult(TASK_BIT, answer, "", raw_trace, False, "compressed_trace_boxed_residue", "low")
    return ReasonerResult(TASK_BIT, answer, compressed, raw_trace, True, "", "high")


def reason_cipher_v0_2(problem):
    """Cipher reasoner with enriched trace (mapping display)."""
    from stage3_reasoners.cipher import reasoning_cipher

    try:
        raw_trace = reasoning_cipher(problem)
    except Exception as exc:
        return ReasonerResult(TASK_CIPHER, "", "", "", False, f"reasoner_error:{exc}", "low")
    if not raw_trace:
        return ReasonerResult(TASK_CIPHER, "", "", "", False, "reasoner_returned_none", "low")

    answer = _extract_reasoner_answer(raw_trace).strip()
    if not answer:
        return ReasonerResult(TASK_CIPHER, "", "", raw_trace, False, "missing_reasoner_answer", "low")

    compressed = build_cipher_trace_v0_2(problem, answer, raw_trace)
    if "\\boxed" in compressed:
        return ReasonerResult(TASK_CIPHER, answer, "", raw_trace, False, "compressed_trace_boxed_residue", "low")
    return ReasonerResult(TASK_CIPHER, answer, compressed, raw_trace, True, "", "high")


def reason_numeral_v0_2(problem):
    """Numeral reasoner with enriched trace (rule display)."""
    from stage3_reasoners.numeral import reasoning_numeral

    try:
        raw_trace = reasoning_numeral(problem)
    except Exception as exc:
        return ReasonerResult(TASK_NUMERAL, "", "", "", False, f"reasoner_error:{exc}", "low")
    if not raw_trace:
        return ReasonerResult(TASK_NUMERAL, "", "", "", False, "reasoner_returned_none", "low")

    answer = _extract_reasoner_answer(raw_trace).strip()
    if not answer:
        return ReasonerResult(TASK_NUMERAL, "", "", raw_trace, False, "missing_reasoner_answer", "low")

    compressed = build_numeral_trace_v0_2(problem, answer, raw_trace)
    if "\\boxed" in compressed:
        return ReasonerResult(TASK_NUMERAL, answer, "", raw_trace, False, "compressed_trace_boxed_residue", "low")
    return ReasonerResult(TASK_NUMERAL, answer, compressed, raw_trace, True, "", "high")


def reason_symbolic_v0_2(problem):
    """Symbolic reasoner — if no rule, returns ok=False (degrades to answer-only)."""
    from stage3_reasoners.cryptarithm import reasoning_cryptarithm
    from stage3_reasoners.equation_numeric import reasoning_equation_numeric

    if EQUATION_NUMERIC_RE.fullmatch(problem.question):
        attempts = (reasoning_equation_numeric, reasoning_cryptarithm)
    else:
        attempts = (reasoning_cryptarithm, reasoning_equation_numeric)

    errors = []
    for fn in attempts:
        try:
            raw_trace = fn(problem)
        except Exception as exc:
            errors.append(f"reasoner_error:{exc}")
            continue
        if not raw_trace:
            errors.append("reasoner_returned_none")
            continue

        answer = _extract_reasoner_answer(raw_trace).strip()
        if not answer:
            errors.append("missing_reasoner_answer")
            continue

        compressed = build_symbolic_trace_v0_2(problem, answer, raw_trace)
        if not compressed:
            return ReasonerResult(TASK_SYMBOLIC, answer, "", raw_trace, False, "no_specific_rule_for_cot", "low")

        if "\\boxed" in compressed:
            errors.append("compressed_trace_boxed_residue")
            continue

        return ReasonerResult(TASK_SYMBOLIC, answer, compressed, raw_trace, True, "", "high")

    return ReasonerResult(TASK_SYMBOLIC, "", "", "", False, "symbolic_reasoners_failed:" + ";".join(errors), "low")


def build_v0_2_registry():
    return ReasonerRegistry({
        TASK_BIT: reason_bit_v0_2,
        TASK_NUMERAL: reason_numeral_v0_2,
        TASK_UNIT: reason_unit_v0_2,
        TASK_GRAVITY: reason_gravity_v0_2,
        TASK_CIPHER: reason_cipher_v0_2,
        TASK_SYMBOLIC: reason_symbolic_v0_2,
    })


# ═══════════════════════════════════════════════════════════════════════
# Ratio control: downsample answer_only to achieve target ratios
# ═══════════════════════════════════════════════════════════════════════

def adjust_sample_ratios(rows, target_cot_ratio=0.475, seed=42):
    """Downsample answer_only to achieve ~target_cot_ratio compressed_cot.

    Current: answer_only ≈ compressed_cot ≈ 50% each (minus replays)
    Target: compressed_cot ~47.5%, answer_only ~32.5%, replays ~20%
    """
    import random
    rng = random.Random(seed)

    answer_only = [r for r in rows if r["sample_type"] == SAMPLE_ANSWER_ONLY]
    compressed_cot = [r for r in rows if r["sample_type"] == SAMPLE_COMPRESSED_COT]
    replays = [r for r in rows if r["sample_type"] in REPLAY_SAMPLE_TYPES]

    total = len(rows)
    n_replays = len(replays)
    n_cot = len(compressed_cot)

    # Target: cot / total ≈ target_cot_ratio
    # cot / (n_ao_keep + cot + replays) = target_cot_ratio
    # n_ao_keep = cot / target_cot_ratio - cot - replays
    target_ao = int(n_cot / target_cot_ratio - n_cot - n_replays) if target_cot_ratio > 0 else 0
    target_ao = max(0, min(target_ao, len(answer_only)))

    if target_ao < len(answer_only):
        rng.shuffle(answer_only)
        answer_only = answer_only[:target_ao]

    result = answer_only + compressed_cot + replays
    rng.shuffle(result)
    return result


# ═══════════════════════════════════════════════════════════════════════
# Custom make_cot_record that stores intermediates
# ═══════════════════════════════════════════════════════════════════════

def make_cot_record_v0_2(problem, result, index, token_counter):
    """Make CoT record with enriched intermediate metadata."""
    record = _base_wonderland_record(
        record_id=f"stage3-wonderland-compressed_cot-{index:06d}-{problem.id}",
        problem=problem,
        sample_type=SAMPLE_COMPRESSED_COT,
        user=build_cot_prompt(problem),
        assistant=build_thinking_completion(result.compressed_trace, result.answer),
        token_counter=token_counter,
        reasoner_result=result,
    )
    # Add intermediate values to metadata
    if result.metadata:
        record["metadata"]["intermediate_values"] = dict(result.metadata)
    return record


# ═══════════════════════════════════════════════════════════════════════
# Modified wonderland row builder with v0_2 reasoners
# ═══════════════════════════════════════════════════════════════════════

def build_wonderland_rows_v0_2(
    problems, *, split, registry, token_counter, skipped, task_skipped,
    raw_traces, reasoner_stats,
):
    """Same as original but uses v0_2 CoT builder."""
    rows = []
    seen_prompt_hashes = set()
    for index, problem in enumerate(problems):
        task_type = task_type_for_problem(problem)
        answer_row = make_answer_only_record(problem, index, token_counter)
        answer_row["id"] = f"stage3-{split}-wonderland-answer_only-{index:06d}-{problem.id}"
        _append_if_valid(rows, answer_row, skipped=skipped, seen_prompt_hashes=seen_prompt_hashes)

        reasoner_stats[task_type]["attempted"] += 1
        result = registry.run(problem)
        raw_traces.append({
            "source_id": problem.id, "split": split, "task_type": task_type,
            "ok": result.ok, "error": result.error, "answer": result.answer,
            "gold_answer": problem.answer, "raw_trace": result.raw_trace,
            "intermediate_values": result.metadata if result.metadata else {},
        })
        if not result.ok:
            reason = f"cot_{task_type}_{result.error or 'reasoner_failed'}"
            skipped[reason] += 1
            task_skipped[task_type][reason] += 1
            continue
        if result.answer.strip() != problem.answer.strip():
            skipped["cot_answer_mismatch"] += 1
            task_skipped[task_type]["cot_answer_mismatch"] += 1
            continue
        reasoner_stats[task_type]["ok"] += 1
        cot_row = make_cot_record_v0_2(problem, result, index, token_counter)
        cot_row["id"] = f"stage3-{split}-wonderland-compressed_cot-{index:06d}-{problem.id}"
        before_len = len(rows)
        _append_if_valid(rows, cot_row, skipped=skipped, seen_prompt_hashes=seen_prompt_hashes)
        if len(rows) == before_len:
            task_skipped[task_type]["cot_validation_or_duplicate_rejected"] += 1
    return rows


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def generate_v0_2(*, input_csv, split_path, output_dir, seed=42, dev_ratio=0.05,
                   max_stage3_prompts=0, stage1_5_replay_path=None,
                   stage2_replay_path=None, replay_train_limit=200,
                   replay_dev_limit=40, dry_run=False):
    """Full v0_2 pipeline: generate + ratio adjust + audit."""
    import random
    from datetime import datetime

    token_counter = QwenTokenCounter(model_id="models/Qwen3-1.7B-Base", cache_dir=Path(".hf-cache/hub"))
    skipped = Counter()
    task_skipped = defaultdict(Counter)
    reasoner_stats = defaultdict(Counter)
    raw_traces = []
    registry = build_v0_2_registry()

    problems, pool_ids, missing_split_ids = select_wonderland_problems(
        input_csv=input_csv, split_path=split_path,
        max_stage3_prompts=max_stage3_prompts, seed=seed, skipped=skipped,
    )
    train_problems, dev_problems = _partition_train_dev(problems, dev_ratio)

    stage1_5_rows = load_replay_rows(stage1_5_replay_path, sample_type=SAMPLE_STAGE1_5_REPLAY, skipped=skipped)
    stage2_rows = load_replay_rows(stage2_replay_path, sample_type=SAMPLE_STAGE2_REPLAY, skipped=skipped)

    rows_by_split = {"train": [], "dev": []}
    rows_by_split["train"].extend(
        build_wonderland_rows_v0_2(
            train_problems, split="train", registry=registry,
            token_counter=token_counter, skipped=skipped,
            task_skipped=task_skipped, raw_traces=raw_traces,
            reasoner_stats=reasoner_stats,
        )
    )
    rows_by_split["dev"].extend(
        build_wonderland_rows_v0_2(
            dev_problems, split="dev", registry=registry,
            token_counter=token_counter, skipped=skipped,
            task_skipped=task_skipped, raw_traces=raw_traces,
            reasoner_stats=reasoner_stats,
        )
    )

    for split, limit in (("train", replay_train_limit), ("dev", replay_dev_limit)):
        if limit <= 0:
            continue
        rows_by_split[split].extend(
            build_replay_rows_for_split(
                stage1_5_rows, split=split, sample_type=SAMPLE_STAGE1_5_REPLAY,
                limit=limit, seed=seed + (11 if split == "train" else 12),
                token_counter=token_counter, skipped=skipped,
            )
        )
        rows_by_split[split].extend(
            build_replay_rows_for_split(
                stage2_rows, split=split, sample_type=SAMPLE_STAGE2_REPLAY,
                limit=limit, seed=seed + (21 if split == "train" else 22),
                token_counter=token_counter, skipped=skipped,
            )
        )

    # Deduplicate
    for split_name in rows_by_split:
        seen = set()
        deduped = []
        for r in rows_by_split[split_name]:
            h = str(r["metadata"]["prompt_hash"])
            if h in seen:
                skipped["duplicate_prompt_hash_across_replay"] += 1
                continue
            seen.add(h)
            deduped.append(r)
        rows_by_split[split_name] = deduped

    # Ratio adjustment: downsample answer_only in train
    pre_counts = Counter(r["sample_type"] for r in rows_by_split["train"])
    print(f"[v0_2] Pre-adjustment train counts: {dict(pre_counts)}")
    rows_by_split["train"] = adjust_sample_ratios(rows_by_split["train"], target_cot_ratio=0.475, seed=seed + 99)
    post_counts = Counter(r["sample_type"] for r in rows_by_split["train"])
    print(f"[v0_2] Post-adjustment train counts: {dict(post_counts)}")

    # Keep dev unchanged (no ratio adjustment)
    validate_all_splits(rows_by_split)

    # Normalize dev IDs to start from 0 (derived from train index)
    for split_name in rows_by_split:
        for ri, r in enumerate(rows_by_split[split_name]):
            r["sequential_index"] = ri

    report = build_report(
        rows_by_split, skipped=skipped, task_skipped=task_skipped,
        reasoner_stats=reasoner_stats, pool_ids=pool_ids,
        missing_split_ids=missing_split_ids, tokenizer=token_counter,
    )

    # Sample type ratios report
    for split_name in rows_by_split:
        counts = Counter(r["sample_type"] for r in rows_by_split[split_name])
        total = len(rows_by_split[split_name])
        print(f"[v0_2] {split_name}: total={total}, ratios: { {k: f'{v/total:.1%}' for k, v in counts.items()} }")

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl_atomic(output_dir / "train.jsonl", rows_by_split["train"])
        write_jsonl_atomic(output_dir / "dev.jsonl", rows_by_split["dev"])
        write_json(output_dir / "report.json", report)
        write_text_atomic(output_dir / "audit.md", build_audit_md(report))
        write_text_atomic(output_dir / "manual_review.md", build_manual_review_md(rows_by_split, raw_traces, report))
        write_jsonl_atomic(output_dir / "debug" / "raw_traces.jsonl", raw_traces)

        # v0_2 specific: trace quality audit (sample 20 per task)
        _write_trace_quality_report(rows_by_split, output_dir)

    return rows_by_split, report


def _write_trace_quality_report(rows_by_split, output_dir):
    """Write stage3_trace_quality_report.md with v0_2 traces."""
    import random
    rng = random.Random(42)

    train_rows = rows_by_split["train"]
    cot_rows = [r for r in train_rows if r["sample_type"] == SAMPLE_COMPRESSED_COT]
    by_tt = defaultdict(list)
    for r in cot_rows:
        by_tt[r["task_type"]].append(r)

    lines = [
        "# Stage 3 v0_2 Compressed CoT Trace Quality Report",
        "",
        f"Total compressed_cot in train: {len(cot_rows)}",
        "",
        "| Task | Sampled | Numeric Intermediate | Rule Name | Query Sub | Final Answer |",
        "|------|---------|---------------------|-----------|-----------|-------------|",
    ]

    for tt in sorted(by_tt.keys()):
        recs = by_tt[tt]
        sample = rng.sample(recs, min(20, len(recs)))
        ni = rn = qs = fa = 0
        for r in sample:
            assistant = r["messages"][1]["content"]
            ni += _has_numeric_intermediate(assistant, tt)
            rn += _has_rule_name(assistant, tt)
            qs += _has_query_sub(assistant, tt)
            fa += _has_final_answer(assistant)
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
    """Check if trace contains concrete numeric intermediate values."""
    if tt in (TASK_GRAVITY, TASK_UNIT):
        return bool(re.search(r"=\s*\d+\.?\d*", assistant))
    if tt == TASK_CIPHER:
        return bool(re.search(r"\w→\w", assistant))
    if tt == TASK_BIT:
        return bool(re.search(r"b\d:", assistant))
    return False


def _has_rule_name(assistant, tt):
    return bool(re.search(r"Rule:|Mapping:|Selected rules:|g =|ratio", assistant, re.IGNORECASE))


def _has_query_sub(assistant, tt):
    return bool(re.search(r"query|convert|decrypt|apply", assistant, re.IGNORECASE))


def _has_final_answer(assistant):
    return bool(assistant.split("</think>")[-1].strip() if "</think>" in assistant else False)


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


def main():
    parser = argparse.ArgumentParser(description="Generate Stage 3 v0_2 enriched data.")
    parser.add_argument("--input-csv", type=Path, default=Path("data/raw/train.csv"))
    parser.add_argument("--split-path", type=Path, default=Path("splits/wonderland_split_seed42.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/instruction/stage3_wonderland_cold_start_v0_2"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-ratio", type=float, default=0.05)
    parser.add_argument("--max-stage3-prompts", type=int, default=0)
    parser.add_argument("--stage1-5-replay-path", type=Path, default=Path("data/instruction/stage1_5/train.jsonl"))
    parser.add_argument("--stage2-replay-path", type=Path, default=Path("data/instruction/stage2_thinking/train.jsonl"))
    parser.add_argument("--replay-train-limit", type=int, default=200)
    parser.add_argument("--replay-dev-limit", type=int, default=40)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"[v0_2] Generating Stage 3 v0_2 data → {args.output_dir}")
    generate_v0_2(
        input_csv=args.input_csv, split_path=args.split_path,
        output_dir=args.output_dir, seed=args.seed,
        dev_ratio=args.dev_ratio, max_stage3_prompts=args.max_stage3_prompts,
        stage1_5_replay_path=args.stage1_5_replay_path,
        stage2_replay_path=args.stage2_replay_path,
        replay_train_limit=args.replay_train_limit,
        replay_dev_limit=args.replay_dev_limit,
        dry_run=args.dry_run,
    )
    print(f"[v0_2] Done. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
