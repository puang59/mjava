"""Shared helpers for WS-B tests (imported via sys.path in the test modules)."""
import json
import shutil
from pathlib import Path

FIX = Path(__file__).resolve().parent


def make_data_dir(dst: Path, n_val: int = 16) -> Path:
    """Copy the hand-written fixture into dst as a data dir (val = first n_val train records)."""
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("train.jsonl", "repair_train.jsonl"):
        shutil.copyfile(FIX / name, dst / name)
    for src, out in (("train.jsonl", "val.jsonl"), ("repair_train.jsonl", "repair_val.jsonl")):
        lines = (FIX / src).read_text(encoding="utf-8").splitlines()[:n_val]
        (dst / out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dst


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
