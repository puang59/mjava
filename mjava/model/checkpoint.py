"""Checkpoint directory layout: ``model.pt`` (state_dict), ``config.yaml``, ``tokenizer.json``."""

from __future__ import annotations

import shutil
from pathlib import Path

import torch
import yaml

from mjava.model.transformer import ModelConfig, Transformer
from mjava.tokenizer import JavaTokenizer, load_tokenizer


def save_checkpoint(out_dir, model: Transformer, tokenizer_path, extra: dict | None = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "model.pt.tmp"
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, tmp)
    tmp.replace(out / "model.pt")  # atomic: a half-written file never replaces a good one
    cfg = {"model": model.cfg.to_dict()}
    if extra:
        cfg.update(extra)
    with open(out / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    dst = out / "tokenizer.json"
    if Path(tokenizer_path).resolve() != dst.resolve():
        shutil.copyfile(tokenizer_path, dst)
    return out


def load_checkpoint(ckpt_dir, device: str = "cpu") -> tuple[Transformer, JavaTokenizer]:
    d = Path(ckpt_dir)
    with open(d / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model = Transformer(ModelConfig.from_dict(cfg["model"]))
    model.load_state_dict(torch.load(d / "model.pt", map_location="cpu", weights_only=True))
    model.to(device).eval()
    return model, load_tokenizer(d / "tokenizer.json")
