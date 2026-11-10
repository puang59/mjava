"""Collect the results of a training run into one zip (default: results.zip) to send back.

    python scripts/pack_results.py [--root .] [--out results.zip] [--extra path ...]

Includes checkpoints/best, runs/compare.md, runs/plots, per-run metrics and evaluation files, runs/run_all_state.json,
the train_gpu*.log files and CHANGES.md; ``--extra`` adds more files or folders (for example code files that were edited).
The smoke-test outputs (runs/smoke, checkpoints/smoke) and resume states (last.pt) are left out.
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path


def collect(root: Path, extra: list[str]) -> list[Path]:
    picks: list[Path] = []
    for name in ("checkpoints/best", "runs/plots", "runs/compare.md", "runs/run_all_state.json", "CHANGES.md",
                 "train_gpu.log", "train_gpu.err.log", *extra):
        p = root / name
        if p.exists():
            picks.append(p)
    runs = root / "runs"
    if runs.is_dir():
        for d in sorted(runs.iterdir()):
            if d.is_dir() and d.name not in ("smoke", "plots"):
                for f in ("metrics.csv", "metrics.jsonl", "eval.json", "eval.md", "eval_examples.jsonl"):
                    if (d / f).is_file():
                        picks.append(d / f)
    files: list[Path] = []
    for p in picks:
        files += [q for q in sorted(p.rglob("*")) if q.is_file()] if p.is_dir() else [p]
    seen, out = set(), []
    for f in files:
        if f.name == "last.pt" or "smoke" in f.relative_to(root).parts:
            continue
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="results.zip")
    ap.add_argument("--extra", nargs="*", default=[], help="additional files or folders, relative to --root")
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    files = collect(root, a.extra)
    if not (root / "checkpoints" / "best" / "model.pt").is_file():
        print("WARNING: checkpoints/best/model.pt not found: the run has not finished (or failed). Packing what exists.",
              file=sys.stderr)
    if not files:
        print("nothing to pack", file=sys.stderr)
        return 1
    out = Path(a.out)
    if not out.is_absolute():
        out = root / out
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.relative_to(root).as_posix())
            print("  +", f.relative_to(root).as_posix())
    print(f"wrote {out} ({out.stat().st_size / 2**20:.1f} MB, {len(files)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
