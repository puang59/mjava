"""Reinforcement-learning fine-tuning with a Java-compiler reward (self-critical sequence training).

Example::

    python -m mjava.train.rl --config gpu --data data --init-from checkpoints/gpu_s2s \
        --name gpu_rl --max-minutes 60
    python -m mjava.train.rl --config gpu --data data --name gpu_rl --resume --max-minutes 60

Algorithm (REINFORCE with a greedy baseline, "SCST")
    for every batch of generate + repair sources
        y_g  = greedy decode               (no grad)         baseline
        y_s  = num_samples samples / source (temperature, top-k)
        R    = reward(java(y)) from the real compiler (all sampled + greedy strings compiled in one pass)
        adv  = R(y_s) - R(y_g of the same source)
        loss_rl  = - mean_rows( adv * mean_t log p(y_s,t) )      (length-normalised)
        loss_mle = cross-entropy of the reference on the same batch
        loss = lambda_rl * loss_rl + (1 - lambda_rl) * loss_mle

Reward (per decoded string; ``conf`` = spec_conformance(spec, java))
    truncated (no <eos>) / empty / compiler timeout / compiler crash -> 0
    compiles                                                          -> conf
    does not compile                                                  -> 0.3 * conf * max(0, 1 - 0.2 * n_errors)
    no usable spec (see below)                                        -> conf := ``no_spec_conf`` (0.5)
A bare ``class A {}`` compiles but only gets the tiny conformance it earns, so it is never a good policy.

Specs: ``train.common.Pair`` has no spec, so train.jsonl / val.jsonl are indexed by id.  A repair example's id is
``"repair-" + base id`` and is mapped back to the base example's spec; failing that, its reference java is looked
up among the generate examples.  If neither works (or the spec has no checkable element) the example uses the
compile-only fallback above; the fraction is logged as ``no_spec_frac`` and printed at start-up.

Train/eval mode: the sampling and the RL re-scoring pass run in ``model.eval()`` (dropout off) so that the
log-probs are exactly those of the policy that produced the samples.  The MLE term is computed in ``train()``
mode (dropout on, like supervised training) in a separate forward pass.

Outputs (same layout as supervised.py)
    checkpoints/<name>/{model.pt,config.yaml,tokenizer.json}   best checkpoint by greedy validation (see below)
    checkpoints/<name>/last.pt                                 resume state
    runs/<name>/metrics.jsonl, metrics.csv                     train / val logs
The supervised starting point is validated first and written as the initial best checkpoint, so the output
directory is always loadable and never worse (on the selection metric) than the supervised model.
Best = lexicographic max of (compile-and-fully-conformant rate, compile rate, mean conformance) of greedy decoding on
a fixed validation subset.  ``rl:`` config keys: see ``RL_DEFAULTS``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import torch
import torch.nn.functional as F

from mjava.data.conformance import required_elements, spec_conformance
from mjava.interfaces import CompileResult, Diagnostic
from mjava.model.checkpoint import load_checkpoint, save_checkpoint
from mjava.model.decoding import SampleOutput, sample, sequence_log_probs
from mjava.tokenizer import JavaTokenizer
from mjava.train.common import (
    Pair,
    build_pairs,
    collate,
    evaluate_loss,
    load_config,
    lr_at,
    make_batches,
    mix_epoch,
    pick_device,
    read_jsonl,
    set_threads,
)

RL_DEFAULTS: dict = {
    "lr": 1.0e-5,
    "weight_decay": 0.0,
    "warmup_steps": 50,
    "max_steps": 2000,
    "lambda_rl": 0.5,  # weight of the RL term; (1 - lambda_rl) goes to MLE on the same batch
    "num_samples": 2,  # sampled sequences per source
    "temperature": 0.8,  # same sampling settings as the retry strategy of the correction loop
    "top_k": 40,
    "batch_examples": 32,  # sources per RL batch
    "batch_tokens": 16000,  # padded reference tokens per batch (len * max_len cap)
    "repair_fraction": 0.3,
    "grad_clip": 1.0,
    "mle_label_smoothing": 0.0,
    "eval_every": 100,
    "log_every": 10,
    "val_gen": 300,  # fixed validation subset: first N generate / M repair examples of val
    "val_fix": 100,
    "val_batch": 32,
    "max_train_examples": None,
    "amp": True,  # CUDA only
    "seed": 0,
    "compile_cache": 8192,  # distinct strings remembered
    "max_chars": 20000,  # longer outputs are not sent to the compiler (reward 0)
    "fail_scale": 0.3,
    "error_penalty": 0.2,
    "no_spec_conf": 0.5,
    "max_oom_skips": 20,
}

SELECT_KEYS = ("full_rate", "compile_rate", "mean_conformance")
CSV_COLS = [
    "step", "split", "loss", "ppl", "loss_rl", "loss_mle", "reward", "reward_greedy", "adv_nonzero_frac",
    "sample_compile", "greedy_compile", "greedy_conf", "no_spec_frac", "lr", "steps_per_s", "elapsed_min",
    "compile_rate", "mean_conformance", "full_rate", "compile_rate_gen", "compile_rate_fix", "full_rate_gen",
    "full_rate_fix", "n",
]


def rl_config(cfg: dict) -> dict:
    rc = dict(RL_DEFAULTS)
    rc.update({k: v for k, v in (cfg.get("rl") or {}).items()})
    unknown = set(cfg.get("rl") or {}) - set(RL_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown rl config keys: {sorted(unknown)} (known: {sorted(RL_DEFAULTS)})")
    for k, dv in RL_DEFAULTS.items():  # PyYAML reads `1e-5` (no dot) as a string; coerce numbers
        v = rc[k]
        if isinstance(dv, bool) or v is None or not isinstance(dv, (int, float)):
            continue
        try:
            rc[k] = int(float(v)) if isinstance(dv, int) else float(v)
        except (TypeError, ValueError):
            raise ValueError(f"rl.{k} must be a number, got {v!r}") from None
    return rc


# ------------------------------------------------------------------------- data
@dataclass
class RLItem(Pair):
    spec: Optional[dict] = None
    spec_via: str = ""  # "id" | "java" | "" (missing)


class SpecIndex:
    """Find the spec of generate examples (by id) and of repair examples (via the base example)."""

    def __init__(self, gen_records: Sequence[dict]):
        self.by_id: dict[str, dict] = {}
        self.by_java: dict[str, dict] = {}
        for r in gen_records:
            sp = r.get("spec")
            if sp and required_elements(sp):
                self.by_id[str(r["id"])] = sp
                self.by_java.setdefault(r["java"], sp)

    @classmethod
    def from_data(cls, data_dir: str | Path, splits: Sequence[str] = ("train", "val")) -> "SpecIndex":
        recs: list[dict] = []
        for s in splits:
            p = Path(data_dir) / f"{s}.jsonl"
            if p.exists():
                recs += read_jsonl(p)
        return cls(recs)

    def lookup(self, rec: dict) -> tuple[Optional[dict], str]:
        rid = str(rec.get("id", ""))
        if rid in self.by_id:
            return self.by_id[rid], "id"
        if rid.startswith("repair-") and rid[len("repair-"):] in self.by_id:
            return self.by_id[rid[len("repair-"):]], "id"
        if rec.get("java") in self.by_java:
            return self.by_java[rec["java"]], "java"
        return None, ""


def load_items(
    tok: JavaTokenizer, data_dir: str | Path, split: str, max_src_len: int, max_tgt_len: int, index: SpecIndex,
    limit_gen: Optional[int] = None, limit_fix: Optional[int] = None,
) -> tuple[list[RLItem], list[RLItem]]:
    """(generate items, repair items) with their specs attached."""
    d = Path(data_dir)
    out = []
    for fname, lim in ((f"{split}.jsonl", limit_gen), (f"repair_{split}.jsonl", limit_fix)):
        p = d / fname
        recs = read_jsonl(p)[:lim] if p.exists() else []
        by_id = {str(r.get("id", "")): r for r in recs}
        pairs, _ = build_pairs(tok, recs, max_src_len, max_tgt_len)
        items = []
        for pr in pairs:
            spec, via = index.lookup(by_id[pr.id])
            items.append(RLItem(pr.src, pr.tgt, pr.kind, pr.id, spec, via))
        out.append(items)
    return out[0], out[1]


# ----------------------------------------------------------------------- reward
@dataclass
class RewardInfo:
    reward: float
    compiled: bool = False
    conformance: float = 0.0
    n_errors: int = 0
    finished: bool = True
    has_spec: bool = True
    failed: bool = False  # timeout / compiler crash


def reward_from_parts(compiled: bool, conf: float, n_errors: int, fail_scale: float = 0.3, error_penalty: float = 0.2) -> float:
    if compiled:
        return float(conf)
    return float(fail_scale * conf * max(0.0, 1.0 - error_penalty * n_errors))


class RewardComputer:
    """Decoded java strings -> rewards, using one JavaCompiler with a small LRU cache of results."""

    def __init__(self, compiler, rc: dict, compiler_factory: Optional[Callable[[], object]] = None):
        self.compiler = compiler
        self.factory = compiler_factory
        self.rc = rc
        self.cache: "OrderedDict[str, CompileResult]" = OrderedDict()
        self.cache_size = int(rc.get("compile_cache", 8192))
        self.compile_calls = 0
        self.cache_hits = 0
        self.crashes = 0
        self.consecutive_bad_batches = 0

    def _compile_all(self, texts: list[str]) -> dict[str, CompileResult]:
        todo = []
        res: dict[str, CompileResult] = {}
        for t in dict.fromkeys(texts):  # unique, ordered
            if t in self.cache:
                self.cache.move_to_end(t)
                res[t] = self.cache[t]
                self.cache_hits += 1
            else:
                todo.append(t)
        if todo:
            self.compile_calls += len(todo)
            try:
                outs = self.compiler.compile_many(todo)
                if len(outs) != len(todo):
                    raise RuntimeError("compile_many returned a wrong number of results")
            except Exception:
                outs = []
                for t in todo:  # isolate the offender
                    try:
                        outs.append(self.compiler.compile(t))
                    except Exception as e:  # noqa: BLE001
                        self.crashes += 1
                        outs.append(CompileResult(False, (Diagnostic("ERROR", -1, -1, "mjava.rl.crash", f"{type(e).__name__}: {e}"),)))
            for t, r in zip(todo, outs):
                res[t] = r
                if not _is_infra_failure(r):  # never cache timeouts / crashes
                    self.cache[t] = r
                    if len(self.cache) > self.cache_size:
                        self.cache.popitem(last=False)
        return res

    def score(self, texts: Sequence[str], finished: Sequence[bool], specs: Sequence[Optional[dict]]) -> list[RewardInfo]:
        rc = self.rc
        max_chars = int(rc["max_chars"])
        need = [t for t, f in zip(texts, finished) if f and t.strip() and len(t) <= max_chars]
        results = self._compile_all(need) if need else {}
        infos = []
        n_infra = 0
        for t, f, sp in zip(texts, finished, specs):
            has_spec = bool(sp) and bool(required_elements(sp))
            if not f or not t.strip() or len(t) > max_chars:
                infos.append(RewardInfo(0.0, finished=bool(f), has_spec=has_spec))
                continue
            r = results[t]
            if _is_infra_failure(r):
                n_infra += 1
                infos.append(RewardInfo(0.0, finished=True, has_spec=has_spec, failed=True))
                continue
            try:
                conf = spec_conformance(sp, t) if has_spec else float(rc["no_spec_conf"])
            except Exception:  # a regex blow-up must not kill training
                conf = 0.0
            ne = len(r.errors)
            infos.append(RewardInfo(
                reward_from_parts(r.ok, conf, ne, rc["fail_scale"], rc["error_penalty"]),
                compiled=bool(r.ok), conformance=conf, n_errors=ne, finished=True, has_spec=has_spec,
            ))
        if need and n_infra == len(need):
            self.consecutive_bad_batches += 1
            if self.factory is not None:  # try a fresh compiler once the old one looks dead
                try:
                    self.compiler.close()
                except Exception:
                    pass
                self.compiler = self.factory()
        elif need:
            self.consecutive_bad_batches = 0
        return infos


def _is_infra_failure(r: CompileResult) -> bool:
    return any(d.code in ("server.timeout", "mjava.rl.crash") for d in r.diagnostics)


# ------------------------------------------------------------------- decoding helpers
def decode_rows(tok: JavaTokenizer, out: SampleOutput) -> tuple[list[str], list[bool]]:
    """SampleOutput -> (java strings, finished flags). Never raises: a bad row decodes to ''."""
    eos = tok.eos_id
    texts, fin = [], out.finished.tolist()
    for row, m in zip(out.tokens.tolist(), out.mask.tolist()):
        ids = [t for t, k in zip(row, m) if k and t != eos]
        try:
            texts.append(tok.decode(ids, skip_special=True))
        except Exception:
            texts.append("")
    return texts, [bool(f) for f in fin]


def max_new_for(items: Sequence[Pair], max_tgt_len: int) -> int:
    longest = max((len(p.tgt) for p in items), default=1)
    return int(min(max_tgt_len, math.ceil(1.5 * longest) + 16))


def compute_advantages(r_sample: Sequence[float], r_greedy: Sequence[float], num_samples: int) -> torch.Tensor:
    g = torch.tensor(list(r_greedy), dtype=torch.float32).repeat_interleave(num_samples)
    return torch.tensor(list(r_sample), dtype=torch.float32) - g


def policy_loss(
    model, src_rep: torch.Tensor, out: SampleOutput, adv: torch.Tensor, temperature: float, top_k: int,
    autocast_ctx=None,
) -> torch.Tensor:
    """-mean_rows(adv * length-normalised log p(sample)); rows with adv == 0 are skipped (no gradient anyway).
    The mean is over ALL rows so that skipping does not rescale the gradient. Gradients flow through
    ``sequence_log_probs`` exactly as in ``sample(..., with_grad=True)``."""
    N = adv.numel()
    sel = (adv != 0).nonzero(as_tuple=True)[0]
    if sel.numel() == 0:
        return torch.zeros((), device=src_rep.device, requires_grad=False)
    dev = src_rep.device
    sel_d = sel.to(dev)
    ctx = autocast_ctx() if autocast_ctx else _null()
    with ctx:
        lp = sequence_log_probs(model, src_rep[sel_d], out.tokens[sel_d], out.mask[sel_d], temperature, top_k)
    lens = out.mask[sel_d].sum(1).clamp_min(1)
    seq_lp = lp.sum(1) / lens
    return -(adv[sel].to(dev) * seq_lp).sum() / N


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


@dataclass
class StepStats:
    loss: float = 0.0
    loss_rl: float = 0.0
    loss_mle: float = 0.0
    reward: float = 0.0
    reward_greedy: float = 0.0
    adv_nonzero_frac: float = 0.0
    sample_compile: float = 0.0
    greedy_compile: float = 0.0
    greedy_conf: float = 0.0
    no_spec_frac: float = 0.0
    n_src: int = 0


def rl_step(
    model, tok: JavaTokenizer, batch_items: Sequence[RLItem], scorer, rc: dict, device: str,
    autocast_ctx=None, scaler=None, optimizer=None, clip: float = 1.0,
) -> StepStats:
    """One SCST + MLE update on a batch. If ``optimizer`` is None only the loss is computed (no backward)."""
    cfg = model.cfg
    pad, bos = cfg.pad_id, cfg.bos_id
    n = int(rc["num_samples"])
    lam = float(rc["lambda_rl"])
    temp, top_k = float(rc["temperature"]), int(rc["top_k"])
    batch = collate(batch_items, pad, bos).to(device)
    specs = [it.spec for it in batch_items]
    max_new = max_new_for(batch_items, cfg.max_tgt_len)
    ctx = autocast_ctx or _null

    # ---- rollouts (eval mode: dropout off => the log-probs are those of the sampling policy)
    model.eval()
    with torch.no_grad(), ctx():
        g_out = sample(model, batch.src, max_new, greedy=True)
        s_out = sample(model, batch.src, max_new, temp, top_k, num_samples=n)
    g_txt, g_fin = decode_rows(tok, g_out)
    s_txt, s_fin = decode_rows(tok, s_out)
    infos = scorer.score(g_txt + s_txt, g_fin + s_fin, specs + [sp for sp in specs for _ in range(n)])
    gi, si = infos[: len(g_txt)], infos[len(g_txt):]
    adv = compute_advantages([i.reward for i in si], [i.reward for i in gi], n)

    # ---- RL term (grad) in eval mode
    src_rep = batch.src.repeat_interleave(n, 0) if n > 1 else batch.src
    l_rl = policy_loss(model, src_rep, s_out, adv.to(batch.src.device), temp, top_k, autocast_ctx)

    # ---- MLE term (grad) in train mode
    model.train()
    ls = float(rc["mle_label_smoothing"])
    with ctx():
        logits = model(batch.src, batch.tgt_in)
    l_mle = F.cross_entropy(logits.float().view(-1, logits.size(-1)), batch.tgt_out.view(-1), ignore_index=pad, label_smoothing=ls)
    loss = lam * l_rl + (1.0 - lam) * l_mle

    if optimizer is not None:
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss (rl={float(l_rl)}, mle={float(l_mle)})")
        optimizer.zero_grad(set_to_none=True)
        (scaler.scale(loss) if scaler is not None else loss).backward()
        if scaler is not None:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

    nb = max(len(gi), 1)
    return StepStats(
        loss=float(loss.detach()), loss_rl=float(l_rl.detach()), loss_mle=float(l_mle.detach()),
        reward=sum(i.reward for i in si) / max(len(si), 1), reward_greedy=sum(i.reward for i in gi) / nb,
        adv_nonzero_frac=float((adv != 0).float().mean()) if adv.numel() else 0.0,
        sample_compile=sum(i.compiled for i in si) / max(len(si), 1), greedy_compile=sum(i.compiled for i in gi) / nb,
        greedy_conf=sum(i.conformance for i in gi) / nb, no_spec_frac=sum(not i.has_spec for i in gi) / nb, n_src=len(gi),
    )


# ------------------------------------------------------------------- validation
def validate_greedy(model, tok: JavaTokenizer, items: Sequence[RLItem], scorer, rc: dict, device: str, autocast_ctx=None) -> dict:
    """Greedy decode every item, compile, and report compile rate / mean conformance / compile-and-fully-conformant rate
    (overall and per kind).  Items without a spec count with the no-spec conformance, so they never count as fully
    conformant unless ``no_spec_conf`` is 1."""
    was = model.training
    model.eval()
    ctx = autocast_ctx or _null
    infos: list[tuple[RLItem, RewardInfo]] = []
    bs = int(rc["val_batch"])
    for b in make_batches(list(items), 10**9, bs, shuffle=False):
        batch = collate(b, model.cfg.pad_id, model.cfg.bos_id).to(device)
        with torch.no_grad(), ctx():
            out = sample(model, batch.src, max_new_for(b, model.cfg.max_tgt_len), greedy=True)
        txt, fin = decode_rows(tok, out)
        infos += list(zip(b, scorer.score(txt, fin, [it.spec for it in b])))
    model.train(was)

    def agg(sel: list[RewardInfo]) -> dict:
        n = max(len(sel), 1)
        return {
            "compile_rate": sum(i.compiled for i in sel) / n,
            "mean_conformance": sum(i.conformance for i in sel) / n,
            "full_rate": sum(i.compiled and i.has_spec and i.conformance >= 1.0 - 1e-9 for i in sel) / n,
            "mean_reward": sum(i.reward for i in sel) / n,
        }

    res = {**agg([i for _, i in infos]), "n": len(infos)}
    for kind in ("gen", "fix"):
        sel = [i for it, i in infos if it.kind == kind]
        if sel:
            a = agg(sel)
            res[f"compile_rate_{kind}"], res[f"full_rate_{kind}"], res[f"conformance_{kind}"] = a["compile_rate"], a["full_rate"], a["mean_conformance"]
    return res


def select_key(v: dict) -> tuple:
    return tuple(v[k] for k in SELECT_KEYS)


# --------------------------------------------------------------------- logging
class MetricsLog:
    def __init__(self, run_dir: Path, resume: bool):
        run_dir.mkdir(parents=True, exist_ok=True)
        mode = "a" if resume else "w"
        self.jf = open(run_dir / "metrics.jsonl", mode, encoding="utf-8")
        new_csv = not (resume and (run_dir / "metrics.csv").exists())
        self.cf = open(run_dir / "metrics.csv", mode, encoding="utf-8", newline="")
        self.cw = csv.DictWriter(self.cf, fieldnames=CSV_COLS, extrasaction="ignore")
        if new_csv:
            self.cw.writeheader()

    def write(self, rec: dict) -> None:
        self.jf.write(json.dumps(rec) + "\n")
        self.jf.flush()
        self.cw.writerow(rec)
        self.cf.flush()

    def close(self) -> None:
        self.jf.close()
        self.cf.close()


# ------------------------------------------------------------------ start-up checks
def check_startup(init_from: Optional[str], ckpt_dir: Path, data_dir: str | Path, resume: bool) -> None:
    """Fail fast with actionable messages (everything that could otherwise fail hours in)."""
    problems = []
    last = ckpt_dir / "last.pt"
    src = Path(init_from) if init_from else None
    if resume and last.exists():
        src = ckpt_dir
    if src is None:
        problems.append("no starting model: pass --init-from <supervised checkpoint dir> (or --resume with an existing run)")
    else:
        for f in ("model.pt", "config.yaml", "tokenizer.json"):
            if not (src / f).exists():
                problems.append(f"{src / f} not found (train the supervised model first: python -m mjava.train.supervised ...)")
    if init_from and Path(init_from).resolve() == ckpt_dir.resolve():
        problems.append("--name must differ from the --init-from directory (RL would overwrite the supervised model)")
    d = Path(data_dir)
    if not (d / "train.jsonl").exists():
        problems.append(f"{d / 'train.jsonl'} not found (build it: python -m mjava.data.build --out {d})")
    if not (d / "val.jsonl").exists():
        problems.append(f"{d / 'val.jsonl'} not found (python -m mjava.data.build --out {d})")
    if shutil.which("javac") is None or shutil.which("java") is None:
        problems.append("javac/java not found on PATH: install a JDK (11 or newer)")
    if problems:
        raise SystemExit("RL start-up check failed:\n  - " + "\n  - ".join(problems))


def make_compiler(kind: str = "auto"):
    from mjava.compiler import get_compiler

    c = get_compiler(kind)
    try:
        r = c.compile("class SmokeTest { int f() { return 1; } }")
        bad = c.compile("class SmokeTest { int f() { return 1 } }")
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"Java compiler smoke test crashed: {e!r}") from e
    if not r.ok or bad.ok:
        raise SystemExit(f"Java compiler smoke test gave wrong answers (ok={r.ok}, broken_ok={bad.ok}); is the JDK working?")
    return c


# ------------------------------------------------------------------------ main loop
def train_rl(
    cfg: dict,
    *,
    data_dir: str = "data",
    init_from: Optional[str] = None,
    name: str = "rl",
    ckpt_root: str = "checkpoints",
    runs_root: str = "runs",
    resume: bool = False,
    max_minutes: Optional[float] = None,
    max_steps: Optional[int] = None,
    threads: Optional[int] = None,
    device: str = "auto",
    compiler=None,
    compiler_kind: str = "auto",
    scorer=None,
    log: Callable[[str], None] = print,
) -> dict:
    """Run RL fine-tuning. Returns a summary dict (baseline/best validation, steps, stop reason).
    ``scorer`` (objects with ``score(texts, finished, specs) -> list[RewardInfo]``) can be injected for tests."""
    rc = rl_config(cfg)
    if max_steps is not None:
        rc["max_steps"] = max_steps
    if rc["num_samples"] < 1:
        raise ValueError("rl.num_samples must be >= 1")
    if not 0.0 <= rc["lambda_rl"] <= 1.0:
        raise ValueError("rl.lambda_rl must be in [0, 1]")
    ckpt_dir = Path(ckpt_root) / name
    run_dir = Path(runs_root) / name
    last_path = ckpt_dir / "last.pt"
    resuming = resume and last_path.exists()
    if scorer is None:
        check_startup(init_from, ckpt_dir, data_dir, resume)
    seed = int(rc["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    device = pick_device(device)
    n_threads = set_threads(threads) if (threads or device == "cpu") else torch.get_num_threads()

    model_dir = ckpt_dir if resuming else Path(init_from or ckpt_dir)
    if not (model_dir / "model.pt").exists():
        raise SystemExit(f"{model_dir / 'model.pt'} not found; pass --init-from <supervised checkpoint dir>")
    try:
        model, tok = load_checkpoint(model_dir, "cpu")
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"cannot load checkpoint {model_dir}: {e!r}") from e
    mcfg = model.cfg
    tok_path = model_dir / "tokenizer.json"
    model.to(device).train()
    log(f"RL device={device} threads={n_threads} params={model.num_parameters():,} init={model_dir}")

    # ---- data
    t0 = time.time()
    index = SpecIndex.from_data(data_dir)
    train_gen, train_fix = load_items(tok, data_dir, "train", mcfg.max_src_len, mcfg.max_tgt_len, index,
                                      rc["max_train_examples"], rc["max_train_examples"])
    val_gen, val_fix = load_items(tok, data_dir, "val", mcfg.max_src_len, mcfg.max_tgt_len, index, int(rc["val_gen"]), int(rc["val_fix"]))
    val_items = val_gen + val_fix
    if not train_gen and not train_fix:
        raise SystemExit(f"no training examples in {data_dir}")
    if not val_items:
        val_items = (train_gen + train_fix)[:100]
        log("warning: no validation data; validating on a slice of train")
    allit = train_gen + train_fix
    miss = sum(1 for it in allit if it.spec is None)
    miss_fix = sum(1 for it in train_fix if it.spec is None)
    log(f"data: {len(train_gen)} gen + {len(train_fix)} repair train, {len(val_items)} val ({time.time() - t0:.1f}s); "
        f"specs missing: {miss}/{len(allit)} train items ({miss_fix}/{len(train_fix)} repair) -> compile-only reward for those")
    if allit and miss / len(allit) > 0.5:
        log("WARNING: most examples have no spec, reward degenerates towards compile-only. Is train.jsonl from mjava.data.build?")

    own_compiler = compiler is None and scorer is None
    if scorer is None:
        if compiler is None:
            compiler = make_compiler(compiler_kind)
            if type(compiler).__name__ == "SubprocessCompiler":
                log("WARNING: using the slow subprocess javac fallback (the persistent compile server failed to start)")
        scorer = RewardComputer(compiler, rc, (lambda: make_compiler(compiler_kind)) if own_compiler else None)

    use_amp = bool(rc["amp"]) and device.startswith("cuda")
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    def autocast_ctx():
        return torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp)

    decay, no_decay = [], []
    for n_, p in model.named_parameters():
        (decay if p.ndim >= 2 and "embed" not in n_ and "_pos" not in n_ else no_decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": rc["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}],
        lr=rc["lr"], betas=(0.9, 0.98),
    )
    step, epoch, batch_in_epoch, elapsed_prev, best_key = 0, 0, 0, 0.0, None
    baseline: Optional[dict] = None
    if resuming:
        st = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        if st.get("scaler"):
            scaler.load_state_dict(st["scaler"])
        step, epoch, batch_in_epoch = st["step"], st["epoch"], st["batch_in_epoch"]
        elapsed_prev, best_key, baseline = st["elapsed"], st["best_key"], st.get("baseline")
        log(f"resumed from {last_path}: step={step} epoch={epoch} elapsed={elapsed_prev / 60:.1f}min best={best_key}")
    elif resume:
        log("--resume given but no last.pt found; starting fresh")

    budget = max_minutes * 60 if max_minutes else None
    total_steps = int(rc["max_steps"])
    metrics = MetricsLog(run_dir, resuming)
    start = time.time()

    def elapsed() -> float:
        return elapsed_prev + time.time() - start

    def save_last() -> None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        tmp = ckpt_dir / "last.pt.tmp"
        torch.save(
            {"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(), "step": step,
             "epoch": epoch, "batch_in_epoch": batch_in_epoch, "elapsed": elapsed(), "best_key": best_key,
             "baseline": baseline},
            tmp,
        )
        tmp.replace(last_path)

    last_val: dict = {}

    def validate(tag: str = "") -> dict:
        nonlocal best_key, last_val
        v = validate_greedy(model, tok, val_items, scorer, rc, device, autocast_ctx)
        lm = evaluate_loss(model, val_items, device)  # teacher-forced loss / ppl on the same subset
        rec = {"step": step, "split": "val", "elapsed_min": elapsed() / 60, "loss": lm["loss"], "ppl": lm["ppl"], **v}
        metrics.write(rec)
        key = select_key(v)
        improved = best_key is None or tuple(key) > tuple(best_key)
        if improved:
            best_key = list(key)
            save_checkpoint(ckpt_dir, model, tok_path, {"stage": "rl", "rl": rc, "step": step, **{f"val_{k}": v[k] for k in SELECT_KEYS}})
        log(f"[val{tag}] step {step} compile {v['compile_rate']:.3f} conf {v['mean_conformance']:.3f} "
            f"full {v['full_rate']:.3f} ppl {lm['ppl']:.3f} n={v['n']} {'(best, saved)' if improved else ''}")
        last_val = v
        return v

    stop_reason = "max_steps"
    oom_skips = 0
    try:
        if not resuming:
            baseline = validate(" baseline")  # supervised model before any update; saved as the initial best
            save_last()
        win = {k: 0.0 for k in StepStats.__dataclass_fields__}
        win_steps, win_t = 0, time.time()
        done = False
        while not done:
            ep_rng = random.Random(seed * 100003 + epoch)
            data = mix_epoch(train_gen, train_fix, float(rc["repair_fraction"]), ep_rng)
            batches = make_batches(data, int(rc["batch_tokens"]), int(rc["batch_examples"]), ep_rng)
            for bi in range(batch_in_epoch, len(batches)):
                if step >= total_steps:
                    done, stop_reason = True, "max_steps"
                    break
                if budget is not None and elapsed() >= budget:
                    done, stop_reason = True, "max_minutes"
                    break
                prog = max(step / total_steps, elapsed() / budget if budget else 0.0)
                lr = lr_at(prog, step, float(rc["lr"]), int(rc["warmup_steps"]))
                for g in opt.param_groups:
                    g["lr"] = lr
                try:
                    s = rl_step(model, tok, batches[bi], scorer, rc, device, autocast_ctx, scaler, opt, float(rc["grad_clip"]))
                except torch.cuda.OutOfMemoryError:
                    oom_skips += 1
                    opt.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    log(f"CUDA OOM at step {step}: skipping batch ({oom_skips}/{rc['max_oom_skips']}); lower rl.batch_tokens/batch_examples")
                    if oom_skips >= int(rc["max_oom_skips"]):
                        raise
                    batch_in_epoch = bi + 1
                    continue
                except FloatingPointError as e:
                    log(f"step {step}: {e}; batch skipped")
                    opt.zero_grad(set_to_none=True)
                    batch_in_epoch = bi + 1
                    continue
                if isinstance(scorer, RewardComputer) and scorer.consecutive_bad_batches >= 5:
                    raise RuntimeError("the Java compiler failed on 5 consecutive batches (timeouts/crashes); aborting. "
                                       "Last state is saved; fix the JDK and --resume.")
                step += 1
                batch_in_epoch = bi + 1
                for k in win:
                    win[k] += getattr(s, k)
                win_steps += 1
                if step % int(rc["log_every"]) == 0:
                    dt = time.time() - win_t
                    rec = {"step": step, "split": "train", "lr": lr, "steps_per_s": win_steps / max(dt, 1e-9),
                           "elapsed_min": elapsed() / 60, **{k: v / win_steps for k, v in win.items() if k != "n_src"}}
                    metrics.write(rec)
                    log(f"step {step} loss {rec['loss']:.4f} (rl {rec['loss_rl']:.4f} mle {rec['loss_mle']:.4f}) "
                        f"R {rec['reward']:.3f} Rg {rec['reward_greedy']:.3f} adv!=0 {rec['adv_nonzero_frac']:.2f} "
                        f"greedy-compile {rec['greedy_compile']:.2f} conf {rec['greedy_conf']:.2f} "
                        f"no-spec {rec['no_spec_frac']:.2f} lr {lr:.1e} {rec['steps_per_s']:.2f} step/s")
                    win = {k: 0.0 for k in win}
                    win_steps, win_t = 0, time.time()
                if step % int(rc["eval_every"]) == 0:
                    validate()
                    save_last()
                    win_t = time.time()
            else:
                epoch += 1
                batch_in_epoch = 0
    except KeyboardInterrupt:
        stop_reason = "interrupted"
        log("interrupted; saving state")
    finally:
        # whatever happened: record the final state so the run can be resumed
        try:
            save_last()
        except Exception as e:  # noqa: BLE001
            log(f"could not write last.pt: {e!r}")
    final = validate(" final")
    save_last()
    metrics.close()
    if own_compiler:
        try:
            scorer.compiler.close()
        except Exception:
            pass
    summary = {
        "steps": step, "stop_reason": stop_reason, "checkpoint": str(ckpt_dir), "minutes": elapsed() / 60,
        "baseline": baseline, "last_val": final, "best": dict(zip(SELECT_KEYS, best_key)) if best_key else None,
        "oom_skips": oom_skips,
    }
    log(f"done: {summary}")
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="RL fine-tuning with a Java compiler reward (self-critical).")
    ap.add_argument("--config", default="gpu", help="tiny|cpu|gpu or a yaml path (the `rl:` section is optional)")
    ap.add_argument("--set", action="append", default=[], help="override, e.g. rl.lr=2e-5 (repeatable)")
    ap.add_argument("--data", default="data")
    ap.add_argument("--init-from", default=None, help="supervised checkpoint dir to start from")
    ap.add_argument("--name", default=None, help="run name (default: <config>_rl)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--device", default="auto", help="auto (cuda if available) | cpu | cuda")
    ap.add_argument("--no-amp", action="store_true", help="disable mixed precision on CUDA")
    ap.add_argument("--compiler", default="auto", choices=["auto", "server", "subprocess"])
    ap.add_argument("--ckpt-root", default="checkpoints")
    ap.add_argument("--runs-root", default="runs")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set + (["rl.amp=false"] if a.no_amp else []))
    train_rl(
        cfg, data_dir=a.data, init_from=a.init_from, name=a.name or f"{a.config}_rl", ckpt_root=a.ckpt_root,
        runs_root=a.runs_root, resume=a.resume, max_minutes=a.max_minutes, max_steps=a.max_steps,
        threads=a.threads, device=a.device, compiler_kind=a.compiler,
    )


if __name__ == "__main__":
    main()
