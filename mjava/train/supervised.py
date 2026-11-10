"""Supervised training: stage ``lm`` (decoder-only pretraining) and ``seq2seq``
(generate + repair mixed, label-smoothed cross-entropy).

Examples::

    python -m mjava.train.supervised --stage seq2seq --config cpu --data data \
        --tokenizer checkpoints/tokenizer.json --name cpu_s2s --max-minutes 60
    python -m mjava.train.supervised --stage seq2seq --config cpu --name cpu_s2s --resume

Outputs
    checkpoints/<name>/{model.pt,config.yaml,tokenizer.json}   best checkpoint (by val loss)
    checkpoints/<name>/last.pt                                 resume state (model+optimizer+counters)
    runs/<name>/metrics.jsonl, metrics.csv                     train / val logs

LR follows warmup + cosine decay where progress = max(step/max_steps, elapsed/max_minutes),
so a wall-clock budget still ends with a fully decayed schedule.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from mjava.model.checkpoint import load_checkpoint, save_checkpoint
from mjava.model.transformer import ModelConfig, Transformer
from mjava.tokenizer import load_tokenizer
from mjava.train.common import (
    build_lm_pairs,
    collate,
    evaluate_loss,
    load_config,
    load_pairs,
    lr_at,
    make_batches,
    mix_epoch,
    pick_device,
    set_threads,
)

CSV_COLS = ["step", "split", "epoch", "loss", "ppl", "ppl_gen", "ppl_fix", "lr", "tok_per_s", "steps_per_s", "elapsed_min"]


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


def _param_groups(model: torch.nn.Module, wd: float):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (decay if p.ndim >= 2 and "embed" not in n and "_pos" not in n else no_decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def train(
    cfg: dict,
    *,
    stage: str = "seq2seq",
    data_dir: str = "data",
    tokenizer_path: Optional[str] = None,
    name: str = "run",
    ckpt_root: str = "checkpoints",
    runs_root: str = "runs",
    init_from: Optional[str] = None,
    resume: bool = False,
    max_minutes: Optional[float] = None,
    max_steps: Optional[int] = None,
    threads: Optional[int] = None,
    device: str = "auto",
    log: Callable[[str], None] = print,
) -> dict:
    """Run supervised training. Returns a summary dict (best val loss/ppl, steps, stop reason)."""
    tc = dict(cfg.get("train", {}))
    if max_steps is not None:
        tc["max_steps"] = max_steps
    if max_minutes is None:
        max_minutes = tc.get("max_minutes")
    seed = int(tc.get("seed", 0))
    random.seed(seed)
    torch.manual_seed(seed)
    device = pick_device(device)
    n_threads = set_threads(threads or tc.get("threads")) if (threads or device == "cpu") else torch.get_num_threads()
    ckpt_dir = Path(ckpt_root) / name
    run_dir = Path(runs_root) / name
    last_path = ckpt_dir / "last.pt"

    # ---- tokenizer / model
    tok_path = Path(tokenizer_path) if tokenizer_path else None
    if resume and last_path.exists():
        tok_path = ckpt_dir / "tokenizer.json"
    if tok_path is None and init_from:
        tok_path = Path(init_from) / "tokenizer.json"
    if tok_path is None or not tok_path.exists():
        raise FileNotFoundError("tokenizer not found; pass --tokenizer (train one with python -m mjava.tokenizer)")
    tok = load_tokenizer(tok_path)

    if init_from:
        model, _ = load_checkpoint(init_from, "cpu")
        mcfg = model.cfg
    else:
        mc = dict(cfg["model"])
        mc.update(vocab_size=tok.vocab_size, pad_id=tok.pad_id, bos_id=tok.bos_id, eos_id=tok.eos_id)
        mcfg = ModelConfig.from_dict(mc)
        model = Transformer(mcfg)
    model.to(device).train()
    log(f"stage={stage} device={device} threads={n_threads} params={model.num_parameters():,}")

    # ---- data
    t0 = time.time()
    if stage == "lm":
        train_gen = build_lm_pairs(tok, data_dir, "train", mcfg.max_src_len, mcfg.max_tgt_len, tc.get("max_train_examples"))
        train_fix = []
        val_pairs = build_lm_pairs(tok, data_dir, "val", mcfg.max_src_len, mcfg.max_tgt_len, tc.get("val_examples", 500))
        val_groups = {"gen": val_pairs}
    elif stage == "seq2seq":
        train_gen, train_fix = load_pairs(
            tok, data_dir, "train", mcfg.max_src_len, mcfg.max_tgt_len, tc.get("max_train_examples")
        )
        n_val = int(tc.get("val_examples", 500))
        rf = float(tc.get("repair_fraction", 0.3))
        vg, vf = load_pairs(tok, data_dir, "val", mcfg.max_src_len, mcfg.max_tgt_len, n_val, int(n_val * rf / (1 - rf)))
        val_pairs = vg + vf
        val_groups = {"gen": vg, "fix": vf}
    else:
        raise ValueError(f"unknown stage {stage}")
    if not train_gen and not train_fix:
        raise RuntimeError("no training examples found")
    if not val_pairs:  # no validation data: validate on a slice of train
        val_pairs = (train_gen + train_fix)[:200]
        val_groups = {"gen": val_pairs}
    log(f"data: {len(train_gen)} gen + {len(train_fix)} repair train, {len(val_pairs)} val ({time.time() - t0:.1f}s tokenising)")

    # ---- optimiser / state
    use_amp = bool(tc.get("amp", False)) and device.startswith("cuda")
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    opt = torch.optim.AdamW(
        _param_groups(model, tc.get("weight_decay", 0.01)),
        lr=tc.get("lr", 1e-3),
        betas=tuple(tc.get("betas", (0.9, 0.98))),
    )
    step, epoch, batch_in_epoch, elapsed_prev, best_val = 0, 0, 0, 0.0, math.inf
    if resume and last_path.exists():
        st = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        if st.get("scaler"):
            scaler.load_state_dict(st["scaler"])
        step, epoch, batch_in_epoch = st["step"], st["epoch"], st["batch_in_epoch"]
        elapsed_prev, best_val = st["elapsed"], st["best_val"]
        log(f"resumed from {last_path}: step={step} epoch={epoch} elapsed={elapsed_prev / 60:.1f}min best_val={best_val:.4f}")
    elif resume:
        log("--resume given but no last.pt found; starting fresh")

    total_steps = int(tc.get("max_steps", 10**9))
    budget = max_minutes * 60 if max_minutes else None
    warmup = int(tc.get("warmup_steps", 200))
    base_lr = float(tc.get("lr", 1e-3))
    ls = float(tc.get("label_smoothing", 0.1))
    clip = float(tc.get("grad_clip", 1.0))
    eval_every = int(tc.get("eval_every", 500))
    log_every = int(tc.get("log_every", 20))
    batch_tokens = int(tc.get("batch_tokens", 6000))
    max_ex = int(tc.get("max_batch_examples", 64))
    rep_frac = float(tc.get("repair_fraction", 0.3))
    pad, bos = mcfg.pad_id, mcfg.bos_id
    metrics = MetricsLog(run_dir, resume and last_path.exists())

    def save_last() -> None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        tmp = ckpt_dir / "last.pt.tmp"
        torch.save(
            {"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(), "step": step, "epoch": epoch,
             "batch_in_epoch": batch_in_epoch, "elapsed": elapsed_prev + time.time() - start, "best_val": best_val},
            tmp,
        )
        tmp.replace(last_path)

    def validate() -> dict:
        nonlocal best_val
        r = evaluate_loss(model, val_pairs, device)
        for k, grp in val_groups.items():
            if grp and len(val_groups) > 1:
                r[f"ppl_{k}"] = evaluate_loss(model, grp, device)["ppl"]
        rec = {"step": step, "split": "val", "epoch": epoch, "elapsed_min": (elapsed_prev + time.time() - start) / 60, **r}
        metrics.write(rec)
        improved = r["loss"] < best_val
        if improved:
            best_val = r["loss"]
            save_checkpoint(ckpt_dir, model, tok_path, {"stage": stage, "train": tc, "step": step, "val_loss": r["loss"], "val_ppl": r["ppl"]})
        log(f"[val] step {step} loss {r['loss']:.4f} ppl {r['ppl']:.3f} {'(best, saved)' if improved else ''}")
        return r

    start = time.time()
    stop_reason = "max_steps"
    win_loss, win_tok, win_steps, win_t = 0.0, 0, 0, time.time()
    last_val: dict = {}
    try:
        done = False
        while not done:
            ep_rng = random.Random(seed * 100003 + epoch)
            data = mix_epoch(train_gen, train_fix, rep_frac, ep_rng)
            batches = make_batches(data, batch_tokens, max_ex, ep_rng)
            for bi in range(batch_in_epoch, len(batches)):
                elapsed = elapsed_prev + time.time() - start
                if step >= total_steps:
                    done, stop_reason = True, "max_steps"
                    break
                if budget is not None and elapsed >= budget:
                    done, stop_reason = True, "max_minutes"
                    break
                progress = max(step / total_steps if total_steps < 10**9 else 0.0, elapsed / budget if budget else 0.0)
                lr = lr_at(progress, step, base_lr, warmup)
                for g in opt.param_groups:
                    g["lr"] = lr
                batch = collate(batches[bi], pad, bos).to(device)
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    logits = model(batch.src, batch.tgt_in)
                loss = F.cross_entropy(
                    logits.float().view(-1, logits.size(-1)), batch.tgt_out.view(-1),
                    ignore_index=pad, label_smoothing=ls,
                )
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(opt)
                scaler.update()
                step += 1
                batch_in_epoch = bi + 1
                win_loss += loss.item()
                win_tok += batch.n_tokens
                win_steps += 1
                if step % log_every == 0:
                    dt = time.time() - win_t
                    rec = {"step": step, "split": "train", "epoch": epoch, "loss": win_loss / win_steps, "lr": lr,
                           "tok_per_s": win_tok / dt, "steps_per_s": win_steps / dt,
                           "elapsed_min": (elapsed_prev + time.time() - start) / 60}
                    metrics.write(rec)
                    log(f"step {step} ep {epoch} loss {rec['loss']:.4f} lr {lr:.2e} {rec['tok_per_s']:.0f} tgt-tok/s {rec['steps_per_s']:.2f} step/s")
                    win_loss, win_tok, win_steps, win_t = 0.0, 0, 0, time.time()
                if step % eval_every == 0:
                    last_val = validate()
                    save_last()
                    win_t = time.time()
            else:
                epoch += 1
                batch_in_epoch = 0
    except KeyboardInterrupt:
        stop_reason = "interrupted"
        log("interrupted; saving state")
    last_val = validate()
    save_last()
    metrics.close()
    summary = {"steps": step, "epochs": epoch, "best_val_loss": best_val, "best_val_ppl": math.exp(min(best_val, 50)),
               "last_val": last_val, "stop_reason": stop_reason, "checkpoint": str(ckpt_dir),
               "minutes": (elapsed_prev + time.time() - start) / 60}
    log(f"done: {summary}")
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Supervised training (lm | seq2seq).")
    ap.add_argument("--stage", choices=["lm", "seq2seq"], default="seq2seq")
    ap.add_argument("--config", default="cpu", help="tiny|cpu|gpu or a yaml path")
    ap.add_argument("--set", action="append", default=[], help="override, e.g. train.lr=5e-4 (repeatable)")
    ap.add_argument("--data", default="data")
    ap.add_argument("--tokenizer", default="checkpoints/tokenizer.json")
    ap.add_argument("--name", default=None, help="run name (default: <config>_<stage>)")
    ap.add_argument("--init-from", default=None, help="checkpoint dir to start from (e.g. the lm stage)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--device", default="auto", help="auto (cuda if available) | cpu | cuda")
    ap.add_argument("--no-amp", action="store_true", help="disable mixed precision on CUDA")
    ap.add_argument("--ckpt-root", default="checkpoints")
    ap.add_argument("--runs-root", default="runs")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set + (["train.amp=false"] if a.no_amp else []))
    train(
        cfg, stage=a.stage, data_dir=a.data, tokenizer_path=a.tokenizer, name=a.name or f"{a.config}_{a.stage}",
        ckpt_root=a.ckpt_root, runs_root=a.runs_root, init_from=a.init_from, resume=a.resume,
        max_minutes=a.max_minutes, max_steps=a.max_steps, threads=a.threads, device=a.device,
    )


if __name__ == "__main__":
    main()
