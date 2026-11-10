"""Human readable rendering of a correction trace (verbose output and the
``.review.txt`` report)."""

from __future__ import annotations

from mjava.pipeline.correct import Attempt, BlockTrace, CorrectionResult


def _indent(text: str, prefix: str = "      | ") -> str:
    return "\n".join(prefix + l for l in text.split("\n")) if text else prefix + "(empty)"


def format_attempt(trace: BlockTrace, a: Attempt) -> str:
    head = f"  attempt {a.index} [{a.strategy}{'' if not a.escalated_from else ', escalated from ' + a.escalated_from}] ({a.phase} check)"
    lines = []
    if a.index == 0:
        lines.append(f"block {trace.index} (lines {trace.start_line}-{trace.end_line}): {trace.instruction}")
    lines.append(head)
    if a.index > 0:
        err = a.source.split(" <err> ", 1)[-1].split(" <code> ", 1)[0]
        lines.append(f"    compiler error fed back to the model: {err}")
        lines.append("    (model input = <fix> instruction <err> error <code> previous attempt)")
    else:
        lines.append(f"    model input: {a.source}")
    if a.code is None:
        lines.append(f"    no code produced: {a.note}")
    else:
        lines.append("    generated:")
        lines.append(_indent(a.code))
    if a.ok:
        lines.append("    result: OK")
    else:
        lines.append("    result: FAILED")
        for d in a.diagnostics:
            if d["kind"] == "ERROR":
                lines.append(f"      line {d['line']}: {d['message']}")
    return "\n".join(lines)


def format_summary(result: CorrectionResult) -> str:
    out = []
    for b in result.blocks:
        if b.success:
            out.append(f"block {b.index}: OK after {b.retries_used} retr{'y' if b.retries_used == 1 else 'ies'}")
        else:
            out.append(f"block {b.index}: FAILED after {b.retries_used} retries")
    return "\n".join(out)


def format_review(result: CorrectionResult, source_name: str) -> str:
    out = [
        f"MANUAL REVIEW NEEDED: {source_name}",
        f"The generated Java did not compile within {result.max_retries} retries.",
        "",
        "Remaining compiler errors (output file line numbers):",
    ]
    errs = [d for d in result.file_diagnostics if d.kind == "ERROR"]
    out += [f"  line {d.line}: {d.message}" for d in errs] or ["  (none reported)"]
    for b in result.blocks:
        out += ["", f"--- block {b.index}, {source_name} lines {b.start_line}-{b.end_line}: "
                    f"{'ok' if b.success else 'FAILED'} ---", f"instruction: {b.instruction}"]
        last = b.last_coded()
        if last is None:
            out.append("(no code was generated)")
            continue
        out.append(f"last attempt: #{last.index} ({last.strategy})")
        out.append("last generated code:")
        out.append(_indent(last.code or "", "  | "))
        for d in last.diagnostics:
            if d["kind"] == "ERROR":
                out.append(f"  error at code line {d['line']}: {d['message']}")
        hist = ", ".join("#%d:%s:%s" % (a.index, a.strategy, "ok" if a.ok else "fail") for a in b.attempts)
        out.append("attempts: " + hist)
    out += ["", "--- last assembled Java ---", result.java.replace("\r\n", "\n")]
    return "\n".join(out) + "\n"
