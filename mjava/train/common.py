"""Shared training utilities (datasets, batching, LR schedule, loss evaluation).

The RL trainer should reuse: ``load_config``, ``load_pairs`` / ``build_pairs`` (token
pairs for generate+repair examples), ``collate``, ``make_batches``, ``lr_at``,
``evaluate_loss`` and ``set_threads``.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
import yaml

from mjava.interfaces import build_generate_source, build_repair_source
from mjava.model.decoding import pad_sequences
from mjava.model.transformer import Transformer
from mjava.tokenizer import CORPUS_SEP, JavaTokenizer

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


# ---------------------------------------------------------------- config / env
_SCI_FLOAT = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)[eE][+-]?\d+$")


def _coerce(v):
    """PyYAML (YAML 1.1) reads ``1e-3`` (no dot) as a string; turn such scalars into floats, recursively."""
    if isinstance(v, str) and _SCI_FLOAT.match(v.strip()):
        return float(v)
    if isinstance(v, dict):
        return {k: _coerce(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_coerce(x) for x in v]
    return v


def load_config(name_or_path: str | Path, overrides: Sequence[str] = ()) -> dict:
    """``name`` (tiny|cpu|gpu) or a yaml path; ``overrides`` like ``train.lr=1e-3``."""
    p = Path(name_or_path)
    if not p.suffix:
        p = CONFIG_DIR / f"{name_or_path}.yaml"
    with open(p, encoding="utf-8") as f:
        cfg = _coerce(yaml.safe_load(f))
    for ov in overrides:
        key, val = ov.split("=", 1)
        node = cfg
        parts = key.split(".")
        for k in parts[:-1]:
            node = node.setdefault(k, {})
        node[parts[-1]] = _coerce(yaml.safe_load(val))
    return cfg


def default_threads() -> int:
    try:
        import psutil  # type: ignore

        n = psutil.cpu_count(logical=False)
        if n:
            return n
    except Exception:
        pass
    return max(1, (os.cpu_count() or 2) // 2)


def set_threads(n: Optional[int] = None) -> int:
    n = n or default_threads()
    torch.set_num_threads(n)
    return n


def pick_device(device: str = "auto") -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


# ------------------------------------------------------------------- datasets
@dataclass
class Pair:
    src: list  # encoder ids (truncated to max_src_len)
    tgt: list  # decoder targets: java ids + eos (decoder input is [bos] + tgt[:-1])
    kind: str  # "gen" | "fix" | "lm"
    id: str = ""


def read_jsonl(path: str | Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def record_source(rec: dict) -> str:
    """Encoder source text for an Example (generate) or RepairExample (repair) dict."""
    if "broken_java" in rec:
        return build_repair_source(rec["instruction"], rec.get("error_text", ""), rec["broken_java"])
    return build_generate_source(rec["instruction"])


def build_pairs(
    tok: JavaTokenizer, records: Sequence[dict], max_src_len: int, max_tgt_len: int
) -> tuple[list[Pair], int]:
    """Tokenise records -> Pairs. Targets longer than ``max_tgt_len`` are dropped (count returned);
    sources are truncated to ``max_src_len``."""
    if not records:
        return [], 0
    srcs = tok.encode_batch([record_source(r) for r in records])
    tgts = tok.encode_batch([r["java"] for r in records])
    pairs, dropped = [], 0
    for r, s, t in zip(records, srcs, tgts):
        t = t + [tok.eos_id]
        if len(t) > max_tgt_len:
            dropped += 1
            continue
        pairs.append(Pair(s[:max_src_len], t, "fix" if "broken_java" in r else "gen", str(r.get("id", ""))))
    return pairs, dropped


def load_pairs(
    tok: JavaTokenizer, data_dir: str | Path, split: str, max_src_len: int, max_tgt_len: int,
    limit_gen: Optional[int] = None, limit_fix: Optional[int] = None,
) -> tuple[list[Pair], list[Pair]]:
    """(generate pairs, repair pairs) for split in train|val|test. Missing repair file -> []."""
    d = Path(data_dir)
    gen_recs = read_jsonl(d / f"{split}.jsonl")[:limit_gen]
    fix_path = d / f"repair_{split}.jsonl"
    fix_recs = read_jsonl(fix_path)[:limit_fix] if fix_path.exists() else []
    gen, _ = build_pairs(tok, gen_recs, max_src_len, max_tgt_len)
    fix, _ = build_pairs(tok, fix_recs, max_src_len, max_tgt_len)
    return gen, fix


def build_lm_pairs(
    tok: JavaTokenizer, data_dir: str | Path, split: str, max_src_len: int, max_tgt_len: int,
    limit: Optional[int] = None,
) -> list[Pair]:
    """Decoder-only LM data: encoder gets an empty ``<gen>`` source. Uses ``corpus_<split>.txt``
    if present, else the ``java`` field of ``<split>.jsonl``. Long files are cut into windows."""
    d = Path(data_dir)
    corpus = d / f"corpus_{split}.txt"
    if corpus.exists():
        texts = [c.strip("\n") for c in corpus.read_text(encoding="utf-8").split(CORPUS_SEP) if c.strip()]
    else:
        texts = [r["java"] for r in read_jsonl(d / f"{split}.jsonl")]
    texts = texts[:limit]
    src = [tok.gen_id]
    pairs = []
    win = max_tgt_len - 1
    for ids in tok.encode_batch(texts):
        for s in range(0, max(len(ids), 1), win):
            chunk = ids[s : s + win]
            last = s + win >= len(ids)
            pairs.append(Pair(list(src), chunk + ([tok.eos_id] if last else []), "lm"))
    return pairs


def mix_epoch(gen: list[Pair], fix: list[Pair], repair_fraction: float, rng: random.Random) -> list[Pair]:
    """All generate pairs plus repair pairs so that repairs are ``repair_fraction`` of the epoch."""
    if not gen:
        return list(fix)
    if not fix or repair_fraction <= 0:
        return list(gen)
    n_fix = int(round(len(gen) * repair_fraction / (1.0 - repair_fraction)))
    chosen: list[Pair] = []
    while len(chosen) < n_fix:
        chosen += rng.sample(fix, min(len(fix), n_fix - len(chosen)))
    return list(gen) + chosen


# ------------------------------------------------------------------- batching
@dataclass
class Batch:
    src: torch.Tensor  # (B,S)
    tgt_in: torch.Tensor  # (B,T)  [bos, y0, ..]
    tgt_out: torch.Tensor  # (B,T)  [y0, .., eos], pad elsewhere
    n_tokens: int  # non-pad target tokens

    def to(self, device) -> "Batch":
        nb = str(device).startswith("cuda")
        return Batch(self.src.to(device, non_blocking=nb), self.tgt_in.to(device, non_blocking=nb),
                     self.tgt_out.to(device, non_blocking=nb), self.n_tokens)


def collate(pairs: Sequence[Pair], pad_id: int, bos_id: int) -> Batch:
    src = pad_sequences([p.src for p in pairs], pad_id)
    tgt_out = pad_sequences([p.tgt for p in pairs], pad_id)
    tgt_in = pad_sequences([[bos_id] + p.tgt[:-1] for p in pairs], pad_id)
    n = int((tgt_out != pad_id).sum())
    if torch.cuda.is_available():  # pinned memory => async host-to-device copies
        src, tgt_in, tgt_out = src.pin_memory(), tgt_in.pin_memory(), tgt_out.pin_memory()
    return Batch(src, tgt_in, tgt_out, n)


def make_batches(
    pairs: Sequence[Pair], max_tokens: int, max_examples: int = 128,
    rng: Optional[random.Random] = None, shuffle: bool = True,
) -> list[list[Pair]]:
    """Length-bucketed batches: within shuffled mega-chunks sort by target length, then fill
    batches so that ``len(batch) * max_tgt_len_in_batch <= max_tokens`` (padded size)."""
    r = rng or random
    idx = list(range(len(pairs)))
    if shuffle:
        r.shuffle(idx)
    chunk = max(max_examples * 50, 1)
    batches: list[list[Pair]] = []
    for s in range(0, len(idx), chunk):
        part = sorted(idx[s : s + chunk], key=lambda i: (len(pairs[i].tgt), len(pairs[i].src)))
        cur: list[Pair] = []
        for i in part:
            p = pairs[i]
            if cur and ((len(cur) + 1) * len(p.tgt) > max_tokens or len(cur) >= max_examples):
                batches.append(cur)
                cur = []
            cur.append(p)
        if cur:
            batches.append(cur)
    if shuffle:
        r.shuffle(batches)
    return batches


# ------------------------------------------------------------------ schedule
def lr_at(progress: float, step: int, base_lr: float, warmup_steps: int, min_ratio: float = 0.1) -> float:
    """Linear warmup then cosine decay driven by ``progress`` in [0,1]
    (progress = max(step/max_steps, elapsed/time_budget))."""
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    p = min(max(progress, 0.0), 1.0)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p)))


# ---------------------------------------------------------------------- eval
@torch.no_grad()
def evaluate_loss(model: Transformer, pairs: Sequence[Pair], device: str, max_tokens: int = 8000) -> dict:
    """Mean cross-entropy per non-pad target token, NO label smoothing, and perplexity=exp(loss)."""
    was = model.training
    model.eval()
    pad, bos = model.cfg.pad_id, model.cfg.bos_id
    total, count = 0.0, 0
    for b in make_batches(pairs, max_tokens, 128, shuffle=False):
        batch = collate(b, pad, bos).to(device)
        logits = model(batch.src, batch.tgt_in)
        loss = F.cross_entropy(
            logits.float().view(-1, logits.size(-1)), batch.tgt_out.view(-1), ignore_index=pad, reduction="sum"
        )
        total += float(loss)
        count += batch.n_tokens
    model.train(was)
    mean = total / max(count, 1)
    return {"loss": mean, "ppl": math.exp(min(mean, 50)), "tokens": count}
