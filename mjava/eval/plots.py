"""PNG curves from the ``metrics.jsonl`` files written by supervised.py and rl.py.

    python -m mjava.eval.plots --run runs/gpu_s2s --run runs/gpu_rl --out runs/plots

supervised.py rows: {step, split: train|val, loss, ppl (val), ...}
rl.py rows:         train {step, loss (mixed), loss_rl, loss_mle, reward, reward_greedy, adv_nonzero_frac,
                           sample_compile, greedy_compile, greedy_conf, ...}
                    val   {step, loss, ppl, compile_rate, mean_conformance, full_rate, ...}  (step 0 = supervised baseline)
Files written to ``--out`` (default: ``<first run>/plots``); a plot is skipped when none of the runs has its columns:
    loss.png (train + val loss), perplexity.png (val ppl), rl_reward.png (mean sampled / greedy reward, share of
    non-zero advantages), rl_greedy_compile.png (train greedy compile rate / conformance), rl_validation.png
    (val greedy compile rate, mean conformance, compile-and-conformant rate).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = ["#2f6db5", "#d9822b", "#3a9a5b", "#9b59b6", "#c0392b", "#7f8c8d"]


def read_metrics(run_dir: str | Path) -> list[dict]:
    p = Path(run_dir) / "metrics.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found (is --run a runs/<name> directory written by supervised.py or rl.py?)")
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass  # half-written last line of a run that is still going
    return rows


def _series(rows: Sequence[dict], split: str, key: str) -> tuple[list, list]:
    pts = sorted({(r["step"], r[key]) for r in rows if r.get("split") == split and r.get(key) is not None})
    return [p[0] for p in pts], [p[1] for p in pts]


def _plot(path: Path, title: str, ylabel: str, curves: list[tuple[str, list, list, str]], ylim=None, logy=False) -> Optional[Path]:
    curves = [c for c in curves if c[1]]
    if not curves:
        return None
    fig, ax = plt.subplots(figsize=(7, 4), dpi=130)
    for i, (label, xs, ys, style) in enumerate(curves):
        ax.plot(xs, ys, style, color=PALETTE[i % len(PALETTE)], label=label, lw=1.6, marker="o" if len(xs) < 25 else None, ms=3)
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def make_plots(runs: Sequence[str | Path], out: str | Path, labels: Optional[Sequence[str]] = None) -> list[Path]:
    runs = [Path(r) for r in runs]
    labels = list(labels) if labels else [r.name for r in runs]
    data = [read_metrics(r) for r in runs]
    out = Path(out)
    made: list[Optional[Path]] = []

    def curves(split: str, key: str, style: str, suffix: str = "") -> list:
        return [(f"{lab}{suffix}", *_series(rows, split, key), style) for lab, rows in zip(labels, data)]

    made.append(_plot(out / "loss.png", "Loss", "cross-entropy (train: as optimised)",
                      curves("train", "loss", "-", " train") + curves("val", "loss", "--", " val")))
    made.append(_plot(out / "perplexity.png", "Validation perplexity", "perplexity", curves("val", "ppl", "-"), logy=True))
    made.append(_plot(out / "rl_reward.png", "RL reward during training", "reward / fraction",
                      curves("train", "reward", "-", " sampled reward") + curves("train", "reward_greedy", "--", " greedy reward")
                      + curves("train", "adv_nonzero_frac", ":", " non-zero advantage"), ylim=(0, 1.02)))
    made.append(_plot(out / "rl_greedy_compile.png", "Greedy decoding on training batches", "rate",
                      curves("train", "greedy_compile", "-", " compile rate") + curves("train", "greedy_conf", "--", " conformance"),
                      ylim=(0, 1.02)))
    made.append(_plot(out / "rl_validation.png", "Validation, greedy decoding (step 0 = supervised start)", "rate",
                      curves("val", "compile_rate", "-", " compile") + curves("val", "mean_conformance", "--", " conformance")
                      + curves("val", "full_rate", ":", " compile+conformant"), ylim=(0, 1.02)))
    return [m for m in made if m is not None]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Plot training curves from runs/<name>/metrics.jsonl.")
    ap.add_argument("--run", action="append", required=True, help="runs/<name> directory (repeatable)")
    ap.add_argument("--label", action="append", default=None, help="legend label per --run (default: directory name)")
    ap.add_argument("--out", default=None, help="output directory (default: <first run>/plots)")
    a = ap.parse_args(argv)
    if a.label and len(a.label) != len(a.run):
        raise SystemExit("--label must be given once per --run")
    out = a.out or str(Path(a.run[0]) / "plots")
    try:
        files = make_plots(a.run, out, a.label)
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e
    for f in files:
        print(f)
    if not files:
        print("no plottable columns found")


if __name__ == "__main__":
    main()
