"""Markdown tables from one or more ``eval.json`` files.

    python -m mjava.eval.compare runs/retrieval/eval.json runs/gpu_s2s/eval.json runs/gpu_rl/eval.json \
        --labels "retrieval baseline,supervised,supervised+RL" --out runs/compare.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence


def _pct(x: Optional[float]) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def _num(x: Optional[float], d: int = 2) -> str:
    return "-" if x is None else f"{x:.{d}f}"


def _get(d: Optional[dict], *path):
    for p in path:
        if d is None:
            return None
        d = d.get(p) if isinstance(d, dict) else None
    return d


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" if i == 0 else "---:" for i in range(len(header))) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def render_comparison(evals: Sequence[dict], labels: Optional[Sequence[str]] = None) -> str:
    labels = list(labels) if labels else [e.get("name", f"run{i}") for i, e in enumerate(evals)]
    ks = sorted({int(k) for e in evals for k in (_get(e, "correction", "success_at") or {})})
    header = ["system", "n", "ppl", "1st compile", "exact", "conformance", "compile+conform"]
    header += [f"success@{k}" for k in ks] + ["mean attempts", "final conformance", "final compile+conform", "repair compile"]
    rows = []
    for lab, e in zip(labels, evals):
        sa = _get(e, "correction", "success_at") or {}
        row = [lab, str(e.get("n", "-")), _num(_get(e, "perplexity", "ppl"), 3),
               _pct(_get(e, "first_attempt", "compile_rate")), _pct(_get(e, "first_attempt", "exact_match")),
               _pct(_get(e, "first_attempt", "mean_conformance")), _pct(_get(e, "first_attempt", "full_rate"))]
        row += [_pct(sa.get(str(k))) for k in ks]
        row += [_num(_get(e, "correction", "mean_attempts")), _pct(_get(e, "correction", "final_mean_conformance")),
                _pct(_get(e, "correction", "final_full_rate")), _pct(_get(e, "repair", "compile_rate"))]
        rows.append(row)
    parts = ["## Overall", _table(header, rows)]

    fams = sorted({f for e in evals for f in (e.get("by_family") or {})})
    if fams:
        h = ["family (n)"] + [f"{lab}: 1st / final compile+conform" for lab in labels]
        frows = []
        for f in fams:
            n = max((_get(e, "by_family", f, "n") or 0) for e in evals)
            cells = []
            for e in evals:
                g = _get(e, "by_family", f)
                cells.append("-" if not g else f"{_pct(g['first_attempt']['full_rate'])} / {_pct(g['correction']['final_full_rate'])}")
            frows.append([f"{f} ({n})"] + cells)
        parts += ["## By task family (compile-and-conformant rate: first attempt / after the correction loop)", _table(h, frows)]

    if any(e.get("heldout") for e in evals):
        h = ["subset (n)"] + [f"{lab}: 1st compile / 1st conform / final compile+conform" for lab in labels]
        hrows = []
        for key, title in (("seen", "seen identifiers"), ("heldout", "held-out identifiers")):
            n = max((_get(e, "heldout", key, "n") or 0) for e in evals)
            cells = []
            for e in evals:
                g = _get(e, "heldout", key)
                cells.append("-" if not g else f"{_pct(g['first_attempt']['compile_rate'])} / {_pct(g['first_attempt']['mean_conformance'])} / {_pct(g['correction']['final_full_rate'])}")
            hrows.append([f"{title} ({n})"] + cells)
        parts += ["## Held-out identifier subset", _table(h, hrows)]
    notes = [f"- {lab}: {n}" for lab, e in zip(labels, evals) for n in (e.get("notes") or [])]
    if notes:
        parts += ["## Notes", "\n".join(notes)]
    return "\n\n".join(parts) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Compare eval.json files in one markdown table.")
    ap.add_argument("files", nargs="+", help="eval.json files (or run directories containing one)")
    ap.add_argument("--labels", default=None, help="comma separated names, same order as files")
    ap.add_argument("--out", default=None, help="write markdown here (default: stdout only)")
    a = ap.parse_args(argv)
    evals = []
    for f in a.files:
        p = Path(f)
        if p.is_dir():
            p = p / "eval.json"
        if not p.exists():
            raise SystemExit(f"{p} not found (run python -m mjava.eval.evaluate first)")
        evals.append(json.loads(p.read_text(encoding="utf-8")))
    labels = a.labels.split(",") if a.labels else None
    if labels and len(labels) != len(evals):
        raise SystemExit("--labels must have one entry per file")
    md = render_comparison(evals, labels)
    print(md)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
