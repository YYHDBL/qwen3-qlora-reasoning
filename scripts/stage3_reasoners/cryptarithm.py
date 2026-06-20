"""Equation symbolic reasoning generator.

Currently handles concatenation operators only (forward and reverse).
Operates directly on the original symbols without letter assignment.
"""

from __future__ import annotations

from dataclasses import dataclass

from stage3_reasoners.store_types import Problem


@dataclass
class _Ex:
    a: tuple[str, str]
    op: str
    b: tuple[str, str]
    out: str


def _concat_type(exs: list[_Ex]) -> str | None:
    """Return 'fwd' if A1A2B1B2, 'rev' if B1B2A1A2, else None."""
    if all(ex.out == ex.a[0] + ex.a[1] + ex.b[0] + ex.b[1] for ex in exs):
        return "fwd"
    if all(ex.out == ex.b[0] + ex.b[1] + ex.a[0] + ex.a[1] for ex in exs):
        return "rev"
    return None


def _box(s: str) -> str:
    """Wrap each character in 【】 brackets."""
    return "".join(f"【{c}】" for c in s)


def reasoning_cryptarithm(problem: Problem) -> str | None:
    """Generate reasoning for cryptarithm problems."""

    def quote(s: str) -> str:
        return f"【{s}】"

    exs: list[_Ex] = []
    for ex in problem.examples:
        inp = str(ex.input_value)
        if len(inp) != 5:
            return None
        exs.append(
            _Ex(
                a=(inp[0], inp[1]),
                op=inp[2],
                b=(inp[3], inp[4]),
                out=str(ex.output_value),
            )
        )

    q = str(problem.question)
    if len(q) != 5:
        return None
    q_a = (q[0], q[1])
    q_op = q[2]
    q_b = (q[3], q[4])

    # Group by operator
    by_op: dict[str, list[_Ex]] = {}
    for parsed_ex in exs:
        by_op.setdefault(parsed_ex.op, []).append(parsed_ex)

    # Detect concat types for each operator
    concat_types: dict[str, str] = {}
    for op, op_exs in by_op.items():
        ct = _concat_type(op_exs)
        if ct is not None:
            concat_types[op] = ct

    # Check question operator for concatenation type (default to fwd if unknown)
    if q_op in by_op:
        q_ct = _concat_type(by_op[q_op])
        if q_ct is None:
            q_ct = "fwd"
    else:
        q_ct = "fwd"

    if q_ct == "fwd":
        answer = q_a[0] + q_a[1] + q_b[0] + q_b[1]
    else:
        answer = q_b[0] + q_b[1] + q_a[0] + q_a[1]

    # Generate trace
    lines: list[str] = []
    lines.append("We need to infer the transformation rule from the examples.")
    lines.append("I will put my final answer inside \\boxed{}.")
    lines.append("")

    # Show each example with concatenation check
    for ex, ex_parsed in zip(problem.examples, exs):
        orig_inp = str(ex.input_value)
        orig_out = str(ex.output_value)
        lines.append(f"{quote(orig_inp)} = {quote(orig_out)}")
        a0, a1 = quote(ex_parsed.a[0]), quote(ex_parsed.a[1])
        b0, b1 = quote(ex_parsed.b[0]), quote(ex_parsed.b[1])
        op_q = quote(ex_parsed.op)
        out_boxed = _box(orig_out)
        lines.append(f"  input: {a0}{a1}{op_q}{b0}{b1}")
        lines.append(f"  left:{a0}{a1}")
        lines.append(f"  operator: {op_q}")
        lines.append(f"  right:{b0}{b1}")
        lines.append(f"  output: {out_boxed}")

        fwd = ex_parsed.a[0] + ex_parsed.a[1] + ex_parsed.b[0] + ex_parsed.b[1]
        rev = ex_parsed.b[0] + ex_parsed.b[1] + ex_parsed.a[0] + ex_parsed.a[1]
        is_fwd = orig_out == fwd
        is_rev = orig_out == rev

        lines.append(
            f"  concatenation: {_box(fwd)} {'match' if is_fwd else 'mismatch'}"
        )
        lines.append(
            f"  reverse concatenation: {_box(rev)} {'match' if is_rev else 'mismatch'}"
        )

        # Operator line with type
        ct = concat_types.get(ex_parsed.op)
        if ct == "fwd":
            op_type = "concatenation"
        elif ct == "rev":
            op_type = "reverse concatenation"
        else:
            op_type = "unknown"
        lines.append(f"  operator: {quote(ex_parsed.op)}{op_type}")
        lines.append("")

    # Apply to question
    q_op_known = q_op in concat_types
    op_label = "concatenation" if q_ct == "fwd" else "reverse concatenation"

    qa0, qa1 = quote(q_a[0]), quote(q_a[1])
    qb0, qb1 = quote(q_b[0]), quote(q_b[1])
    q_orig = str(problem.question)
    lines.append(f"Question{quote(q_orig)}")
    lines.append(f"  input: {qa0}{qa1}{quote(q_op)}{qb0}{qb1}")
    lines.append(f"  left:{qa0}{qa1}")
    lines.append(f"  operator:{quote(q_op)}")
    lines.append(f"  right:{qb0}{qb1}")
    lines.append("")

    if q_op_known:
        lines.append(
            f"The question operator is {quote(q_op)}, which is {op_label}."
        )
    else:
        lines.append(f"The question operator is {quote(q_op)}, which is unknown.")
        lines.append(
            "As the question operator is unknown, we default to concatenation."
        )
    lines.append("")

    lines.append(
        f"  {op_label}({qa0}{qa1}, {qb0}{qb1}) = {_box(answer)}"
    )
    lines.append(f"  output: {quote(answer)}-> {quote('{' + answer + '}')}")
    lines.append("")
    lines.append("I will now return the answer in \\boxed{}")
    lines.append(f"The answer in \\boxed{{–}} is \\boxed{{{answer}}}")
    return "\n".join(lines)
