"""Equation numeric reasoning generator."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from stage3_reasoners.store_types import Problem

_EXPR_RE = re.compile(r"^(\d+)(\D)(\d+)$")


def _common_candidates(a: int, b: int, sa: str, sb: str) -> list[tuple[str, str]]:
    """Common operations tried first."""
    out: list[tuple[str, str]] = []
    out.append(("concatenation", sa + sb))
    out.append(("reverse concatenation", sb + sa))
    out.append(("addition", str(a + b)))
    out.append(("absolute difference", str(abs(a - b))))
    out.append(("negated absolute difference", str(-abs(a - b))))
    out.append(("subtraction (a-b)", str(a - b)))
    out.append(("reverse subtraction (b-a)", str(b - a)))
    out.append(("multiplication", str(a * b)))
    return out


def _rare_candidates(a: int, b: int, sa: str, sb: str) -> list[tuple[str, str]]:
    """Rare operations tried if common ones don't match."""
    out: list[tuple[str, str]] = []
    out.append(("multiply+1", str(a * b + 1)))
    out.append(("multiply-1", str(a * b - 1)))
    out.append(("add+1", str(a + b + 1)))
    out.append(("add-1", str(a + b - 1)))
    out.append(("sub+1", str(a - b + 1)))
    out.append(("sub-1", str(a - b - 1)))
    if a != 0 and b != 0:
        big, small = max(a, b), min(a, b)
        out.append(("max mod min", str(big % small)))
    if b != 0:
        out.append(("integer division (a/b)", str(a // b)))
        out.append(("modulo (a mod b)", str(a % b)))
    if a != 0:
        out.append(("reverse division (b/a)", str(b // a)))
        out.append(("reverse modulo (b mod a)", str(b % a)))
    if len(sa) == 2 and len(sb) == 2:
        d1, d2, d3, d4 = int(sa[0]), int(sa[1]), int(sb[0]), int(sb[1])
        out.append(("digit absolute diff", str(abs(d1 - d3)) + str(abs(d2 - d4))))
        out.append(("digit add mod10", str((d1 + d3) % 10) + str((d2 + d4) % 10)))
        out.append(("digit sub mod10", str((d1 - d3) % 10) + str((d2 - d4) % 10)))
        out.append(("cross multiply", str(d1 * d3 + d2 * d4)))
        out.append(("cross multiply rev", str(d1 * d4 + d2 * d3)))
        out.append(("digit multiply", str(d1 * d3) + str(d2 * d4)))
        out.append(("digit multiply rev", str(d1 * d4) + str(d2 * d3)))
        out.append(("digit sum diff", str((d1 + d2) - (d3 + d4))))
        out.append(("digit sum sum", str((d1 + d2) + (d3 + d4))))
        out.append(("digit product diff", str(d1 * d2 - d3 * d4)))
        out.append(("digit product sum", str(d1 * d2 + d3 * d4)))
        det_val = d1 * d4 - d2 * d3
        out.append(("determinant", str(det_val)))
        out.append(("abs determinant", str(abs(det_val))))
    return out


def _all_candidates(a: int, b: int, sa: str, sb: str) -> list[tuple[str, str]]:
    """All candidates: common first, then rare."""
    return _common_candidates(a, b, sa, sb) + _rare_candidates(a, b, sa, sb)


def _expr(name: str, a: str, b: str) -> str:
    """Return the math expression for an operation, e.g. '94 + 48'."""
    if name == "addition":
        return f"{a} + {b}"
    if name == "subtraction (a-b)":
        return f"{a} - {b}"
    if name == "reverse subtraction (b-a)":
        return f"{b} - {a}"
    if name == "multiplication":
        if len(a) >= 2:
            decomp = " + ".join(
                str(int(d) * (10 ** (len(a) - 1 - i))) for i, d in enumerate(a)
            )
            return f"({decomp}) * {b}"
        return f"{a} * {b}"
    if name == "absolute difference":
        return f"|{a} - {b}|"
    if name == "negated absolute difference":
        return f"-|{a} - {b}|"
    if name == "concatenation":
        return f"{a} || {b}"
    if name == "reverse concatenation":
        return f"{b} || {a}"
    if name == "multiply+1":
        if len(a) >= 2:
            decomp = " + ".join(
                str(int(d) * (10 ** (len(a) - 1 - i))) for i, d in enumerate(a)
            )
            return f"({decomp}) * {b} + 1"
        return f"{a} * {b} + 1"
    if name == "multiply-1":
        if len(a) >= 2:
            decomp = " + ".join(
                str(int(d) * (10 ** (len(a) - 1 - i))) for i, d in enumerate(a)
            )
            return f"({decomp}) * {b} - 1"
        return f"{a} * {b} - 1"
    if name == "add+1":
        return f"{a} + {b} + 1"
    if name == "add-1":
        return f"{a} + {b} - 1"
    if name == "sub+1":
        return f"{a} - {b} + 1"
    if name == "sub-1":
        return f"{a} - {b} - 1"
    if name == "integer division (a/b)":
        return f"{a} / {b}"
    if name == "modulo (a mod b)":
        return f"{a} mod {b}"
    if name == "reverse division (b/a)":
        return f"{b} / {a}"
    if name == "reverse modulo (b mod a)":
        return f"{b} mod {a}"
    if name == "max mod min":
        big, small = (a, b) if int(a) >= int(b) else (b, a)
        return f"max({a},{b}) mod min({a},{b}) = {big} mod {small}"
    if len(a) == 2 and len(b) == 2:
        d1, d2, d3, d4 = a[0], a[1], b[0], b[1]
        if name == "digit absolute diff":
            return f"|{d1}-{d3}| || |{d2}-{d4}|"
        if name == "digit add mod10":
            return f"({d1}+{d3})%10 || ({d2}+{d4})%10"
        if name == "digit sub mod10":
            return f"({d1}-{d3})%10 || ({d2}-{d4})%10"
        if name == "cross multiply":
            return f"{d1}*{d3} + {d2}*{d4}"
        if name == "cross multiply rev":
            return f"{d1}*{d4} + {d2}*{d3}"
        if name == "digit multiply":
            return f"{d1}*{d3} || {d2}*{d4}"
        if name == "digit multiply rev":
            return f"{d1}*{d4} || {d2}*{d3}"
        if name == "digit sum diff":
            return f"({d1}+{d2}) - ({d3}+{d4})"
        if name == "digit sum sum":
            return f"({d1}+{d2}) + ({d3}+{d4})"
        if name == "digit product diff":
            return f"{d1}*{d2} - {d3}*{d4}"
        if name == "digit product sum":
            return f"{d1}*{d2} + {d3}*{d4}"
        if name == "determinant":
            return f"{d1}*{d4} - {d2}*{d3}"
        if name == "abs determinant":
            return f"|{d1}*{d4} - {d2}*{d3}|"
    return ""


def _expr_intermediate(name: str, a: str, b: str) -> str:
    """Return intermediate evaluated form for operations with multiplications, else ''."""
    ia, ib = int(a), int(b)
    if name in ("multiply+1", "multiply-1", "multiplication") and len(a) >= 2:
        # Decompose a by place value: 70 → [70, 0], 73 → [70, 3]
        places = [int(d) * (10 ** (len(a) - 1 - i)) for i, d in enumerate(a)]
        decomp = " + ".join(f"{p} * {b}" for p in places)
        evald = " + ".join(str(p * ib) for p in places)
        product_sum = sum(p * ib for p in places)
        if name == "multiply+1":
            return f"{decomp} + 1 = {evald} + 1 = {product_sum} + 1"
        if name == "multiply-1":
            return f"{decomp} - 1 = {evald} - 1 = {product_sum} - 1"
        return f"{decomp} = {evald}"
    if len(a) == 2 and len(b) == 2:
        d1, d2, d3, d4 = int(a[0]), int(a[1]), int(b[0]), int(b[1])
        if name == "cross multiply":
            return f"{d1 * d3} + {d2 * d4}"
        if name == "cross multiply rev":
            return f"{d1 * d4} + {d2 * d3}"
        if name == "digit multiply":
            return f"{d1 * d3} || {d2 * d4}"
        if name == "digit multiply rev":
            return f"{d1 * d4} || {d2 * d3}"
        if name == "digit product diff":
            return f"{d1 * d2} - {d3 * d4}"
        if name == "digit product sum":
            return f"{d1 * d2} + {d3 * d4}"
        if name == "determinant":
            return f"{d1 * d4} - {d2 * d3}"
        if name == "abs determinant":
            return f"|{d1 * d4} - {d2 * d3}|"
    return ""


def _rev(s: str) -> str:
    if s.startswith("-"):
        return "-" + s[1:][::-1]
    return s[::-1]


@dataclass
class FoundOp:
    op_name: str
    rev_ops: bool
    rev_res: bool
    fmt: str
    op_char: str


def _apply_op(found: FoundOp, a_str: str, b_str: str) -> tuple[str, list[str]]:
    """Apply the found operation and return (result, explanation_lines)."""
    steps: list[str] = []
    ta = a_str[::-1] if found.rev_ops else a_str
    tb = b_str[::-1] if found.rev_ops else b_str

    # Header line always present
    if found.rev_ops and found.rev_res:
        steps.append(
            f"reversed operands [{a_str}->{ta}, {b_str}->{tb}] and reversed result"
        )
    elif found.rev_ops:
        steps.append(f"reversed operands [{a_str}->{ta}, {b_str}->{tb}]")
    elif found.rev_res:
        steps.append("reversed result")
    else:
        steps.append("identity")

    # Find the matching candidate
    raw_result = ""
    for name, res in _all_candidates(int(ta), int(tb), ta, tb):
        if name == found.op_name:
            raw_result = res
            break

    final = _rev(raw_result) if found.rev_res else raw_result

    expr = _expr(found.op_name, ta, tb)
    inter = _expr_intermediate(found.op_name, ta, tb)
    if expr and inter:
        detail = f" {expr} = {inter} ="
    elif expr:
        detail = f" {expr} ="
    else:
        detail = ""
    val = f"{raw_result} -rev-> {final}" if found.rev_res else final
    steps.append(f"{found.op_name} f({ta}, {tb}) ={detail} {val}")

    if found.fmt == "pre":
        final = found.op_char + final
        steps.append(f"Prefix operator: {final}")
    elif found.fmt == "neg_suffix":
        if final.startswith("-"):
            old = final
            final = final[1:] + found.op_char
            steps.append(
                f"Result is negative - we add back the operator suffix 【{found.op_char}】: {old} -> 【{final}】"
            )
        else:
            steps.append(f"Result is non-negative, no suffix needed: 【{final}】")
    elif found.fmt == "neg_prefix":
        if final.startswith("-"):
            old = final
            final = found.op_char + final[1:]
            steps.append(
                f"Result is negative - we add back the operator prefix 【{found.op_char}】: {old} -> 【{final}】"
            )
        else:
            steps.append(f"Result is non-negative, no prefix needed: 【{final}】")

    return final, steps


def reasoning_equation_numeric(problem: Problem) -> str | None:
    lines: list[str] = []
    lines.append("We need to infer the transformation rule from the examples.")
    lines.append("I will put my final answer inside \\boxed{}.")
    lines.append("")
    lines.append("Examples:")

    parsed: list[tuple[str, str, str, str]] = []
    for ex in problem.examples:
        m = _EXPR_RE.fullmatch(str(ex.input_value))
        if not m:
            continue
        a, op, b = m.group(1), m.group(2), m.group(3)
        parsed.append((a, op, b, str(ex.output_value)))
        lines.append(f"  {ex.input_value} = {ex.output_value}")

    by_op: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for a, op, b, out in parsed:
        by_op[op].append((a, b, out))

    # Precompute prefix/suffix format and transformed groups per operator
    detected_fmts: dict[str, str] = {}
    transformed_groups: dict[str, list[tuple[str, str, str]]] = {}
    has_symbol_suffix = False
    has_symbol_prefix = False
    symbol_suffix_char = ""
    symbol_prefix_char = ""

    for op_char, group in by_op.items():
        any_neg_suffixed = op_char != "-" and any(
            out.endswith("-") and len(out) > 1 for _, _, out in group
        )
        any_neg_prefixed = op_char != "-" and any(
            out.startswith("-") and len(out) > 1 for _, _, out in group
        )
        any_suffixed = any(
            out.endswith(op_char) and len(out) > 1 for _, _, out in group
        )
        any_prefixed = any(
            out.startswith(op_char) and len(out) > 1 for _, _, out in group
        )

        fmt = "num"
        transformed = list(group)
        if any_neg_suffixed:
            fmt = "neg_suffix"
            transformed = [
                (a, b, "-" + out[:-1] if out.endswith("-") and len(out) > 1 else out)
                for a, b, out in group
            ]
        elif any_neg_prefixed:
            fmt = "neg_prefix"
        elif any_suffixed:
            fmt = "neg_suffix"
            has_symbol_suffix = True
            symbol_suffix_char = op_char
            transformed = [
                (
                    a,
                    b,
                    "-" + out[: -len(op_char)]
                    if out.endswith(op_char) and len(out) > 1
                    else out,
                )
                for a, b, out in group
            ]
        elif any_prefixed:
            fmt = "neg_prefix"
            has_symbol_prefix = True
            symbol_prefix_char = op_char
            transformed = [
                (
                    a,
                    b,
                    "-" + out[len(op_char) :]
                    if out.startswith(op_char) and len(out) > 1
                    else out,
                )
                for a, b, out in group
            ]

        detected_fmts[op_char] = fmt
        transformed_groups[op_char] = transformed

    # Build map from (a, op, b) to transformed output
    transformed_map: dict[tuple[str, str, str], str] = {}
    for oc, tgroup in transformed_groups.items():
        for a, b, tout in tgroup:
            transformed_map[(a, oc, b)] = tout

    # Check inputs for leading zeros
    all_inputs: list[str] = []
    for a, _, b, _ in parsed:
        all_inputs.append(a)
        all_inputs.append(b)
    lines.append("")
    lines.append(f"The inputs are {', '.join(all_inputs)}")

    # Report outputs
    all_outputs = [out for _, _, _, out in parsed]
    lines.append("")
    lines.append(f"The outputs are {', '.join(all_outputs)}")
    if has_symbol_suffix:
        lines.append(
            f"Some outputs have the operator symbol as suffix 【{symbol_suffix_char}】."
        )
    if has_symbol_prefix:
        lines.append(
            f"Some outputs have the operator symbol as prefix 【{symbol_prefix_char}】."
        )
    if not has_symbol_suffix and not has_symbol_prefix:
        lines.append("No outputs have a symbol prefix or suffix.")

    # Show transformed outputs if any transformation occurred
    any_transformed = any(fmt != "num" for fmt in detected_fmts.values())
    if any_transformed:
        t_all = [transformed_map.get((a, op, b), out) for a, op, b, out in parsed]
        lines.append(f"We now consider the outputs to be {', '.join(t_all)}")
        if has_symbol_suffix:
            lines.append(
                "We will add back the operator suffix if our answer is negative."
            )
        elif has_symbol_prefix:
            lines.append(
                "We will add back the operator prefix if our answer is negative."
            )

    lines.append("")

    # Show input → operator parsing
    lines.append("Looking at the input of the examples")
    for a, op, b, out in parsed:
        lines.append(f"{a}{op}{b} -> {op}")
    op_names = list(by_op.keys())
    lines.append("")
    lines.append("The operators")
    for op in op_names:
        lines.append(op)

    q_match = _EXPR_RE.fullmatch(str(problem.question))
    q_op = q_match.group(2) if q_match else None

    lines.append("")
    lines.append("Looking at the question")
    if q_match:
        lines.append(f"{problem.question} -> {q_op}")

    # If question operator not in examples, fall back to most common example operator
    effective_q_op = q_op
    if q_op is not None and q_op not in by_op and by_op:
        most_common_op = max(by_op, key=lambda op: len(by_op[op]))
        lines.append(
            f"The question operator is not found in the examples. "
            f"Investigating the most common example operator 【{most_common_op}】 instead. "
            f"We will use absolute difference for the question operator."
        )
        effective_q_op = most_common_op
    elif q_op is not None and q_op in by_op:
        lines.append("The question operator is found in the examples.")

    found_ops: dict[str, FoundOp] = {}

    # Analyze each operator (focus on question operator)
    for op_char, group in sorted(by_op.items()):
        if effective_q_op is not None and op_char != effective_q_op and len(by_op) > 1:
            continue

        # Use precomputed format and transformed group
        detected_fmt = detected_fmts[op_char]
        group = transformed_groups[op_char]

        examples_str = ", ".join(f"{a}{op_char}{b} = {out}" for a, b, out in group)
        lines.append("")
        lines.append(f"Looking at operator 【{op_char}】 [{examples_str}]:")

        a_str, b_str, expected = group[0]

        # Try common operations first (all 4 combos), then rare operations
        found = None

        candidate_sets = [
            ("common", _common_candidates),
            ("rare", _rare_candidates),
        ]

        n_ex = len(group)
        for set_name, cand_fn in candidate_sets:
            for rev_ops, rev_res in (
                (True, True),
                (False, False),
                (True, False),
                (False, True),
            ):
                # Use fixed example order for paragraph header
                cycled = list(group)

                # Describe what we're trying
                label = f"{set_name} operations"
                if rev_ops:
                    rev_parts = ", ".join(
                        f"{ax}->{ax[::-1]} {bx}->{bx[::-1]}" for ax, bx, _ in cycled
                    )
                    if rev_res:
                        label += f" reversed operands [{rev_parts}] and reversed result"
                    else:
                        label += f" reversed operands [{rev_parts}]"
                elif rev_res:
                    id_parts = ", ".join(f"{ax} {bx}" for ax, bx, _ in cycled)
                    label += f" identity operands [{id_parts}] reversed result"
                else:
                    id_parts = ", ".join(f"{ax} {bx}" for ax, bx, _ in cycled)
                    label += f" on identity [{id_parts}]"
                if rev_ops:
                    all_expected = ", ".join(
                        f"({ax[::-1]},{bx[::-1]})->{exp}" for ax, bx, exp in cycled
                    )
                else:
                    all_expected = ", ".join(
                        f"({ax},{bx})->{exp}" for ax, bx, exp in cycled
                    )
                lines.append(f"  Trying {label} [expected {all_expected}]:")

                def _fmt_result(
                    raw: str, a: str, b: str, detail: str, arrow: bool
                ) -> str:
                    fin = _rev(raw) if rev_res else raw
                    val = f"{raw} -rev-> {fin}" if rev_res else fin
                    if arrow:
                        return f"f({a},{b}) ->{detail} {val}"
                    return f"f({a}, {b}) ={detail} {val}"

                # Use first example for candidate generation
                ca_str, cb_str = cycled[0][0], cycled[0][1]
                cta = ca_str[::-1] if rev_ops else ca_str
                ctb = cb_str[::-1] if rev_ops else cb_str
                candidates = cand_fn(int(cta), int(ctb), cta, ctb)
                cand_idx = 0
                for cand_name, cand_res in candidates:
                    # Rotate which example is tried first within the paragraph
                    rotated = [cycled[(cand_idx + j) % n_ex] for j in range(n_ex)]
                    cand_idx += 1

                    parts = []
                    all_pass = True

                    for i, (ax, bx, exp_x) in enumerate(rotated):
                        rax = ax[::-1] if rev_ops else ax
                        rbx = bx[::-1] if rev_ops else bx
                        raw = next(
                            r
                            for n, r in _all_candidates(int(rax), int(rbx), rax, rbx)
                            if n == cand_name
                        )
                        expr_x = _expr(cand_name, rax, rbx)
                        inter_x = _expr_intermediate(cand_name, rax, rbx)
                        if expr_x and inter_x:
                            detail_x = f" {expr_x} = {inter_x} ="
                        elif expr_x:
                            detail_x = f" {expr_x} ="
                        else:
                            detail_x = ""
                        fin = _rev(raw) if rev_res else raw
                        status = "match" if fin == exp_x else "wrong"
                        if fin != exp_x:
                            all_pass = False
                        parts.append(
                            _fmt_result(raw, rax, rbx, detail_x, arrow=i > 0)
                            + f" {status}"
                        )
                        if fin != exp_x:
                            break

                    if all_pass:
                        if found:
                            parts.append("correct, but skipping")
                        else:
                            summary = []
                            if rev_ops:
                                summary.append("reversed operands")
                            if rev_res:
                                summary.append("reversed result")
                            summary.append(cand_name)
                            parts.append("correct, actions: " + ", ".join(summary))
                    lines.append(f"    {cand_name} " + ", ".join(parts))

                    if not all_pass:
                        continue

                    if not found:
                        found = FoundOp(
                            op_name=cand_name,
                            rev_ops=rev_ops,
                            rev_res=rev_res,
                            fmt=detected_fmt,
                            op_char=op_char,
                        )

        if found:
            found_ops[op_char] = found
        else:
            if op_char == effective_q_op:
                return None
            lines.append("  No matching operation found.")

    # Apply to question
    if not q_match or effective_q_op not in found_ops:
        return None

    qa, qb = q_match.group(1), q_match.group(3)
    lines.append("")
    lines.append(f"Applying to {problem.question}:")
    if effective_q_op != q_op:
        lines.append(
            "  We recall that the question operator is not found in the examples. "
            "We will use the absolute difference as the operator."
        )
        abs_diff_op = FoundOp(
            op_name="absolute difference",
            rev_ops=False,
            rev_res=False,
            fmt=found_ops[effective_q_op].fmt,
            op_char=q_op or "",
        )
        result_val, steps = _apply_op(abs_diff_op, qa, qb)
    else:
        result_val, steps = _apply_op(found_ops[effective_q_op], qa, qb)
    for step in steps:
        lines.append(f"  {step}")
    lines.append(f"  Result: 【{result_val}】")

    lines.append("")
    lines.append("I will now return the answer in \\boxed{}")
    lines.append(f"The answer in \\boxed{{–}} is \\boxed{{{result_val}}}")
    return "\n".join(lines)
