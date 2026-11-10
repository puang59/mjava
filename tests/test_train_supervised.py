import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
from helpers import FIX, make_data_dir, read_jsonl  # noqa: E402

from mjava.interfaces import GenerationConfig, build_generate_source  # noqa: E402
from mjava.model import TransformerGenerator  # noqa: E402
from mjava.tokenizer import load_tokenizer, train_tokenizer  # noqa: E402
from mjava.train.common import (  # noqa: E402
    collate, evaluate_loss, load_config, load_pairs, lr_at, make_batches,
)
from mjava.train.supervised import train  # noqa: E402


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("sup")
    data = make_data_dir(root / "data")
    tok_path = root / "tokenizer.json"
    train_tokenizer([data], tok_path, vocab_size=500)
    return root, data, tok_path


def _quiet(_msg):
    pass


def test_batches_are_length_bucketed(env):
    _, data, tok_path = env
    tok = load_tokenizer(tok_path)
    gen, fix = load_pairs(tok, data, "train", 128, 256)
    assert len(gen) == 64 and len(fix) > 0
    batches = make_batches(gen + fix, 1500, 8, shuffle=False)
    assert sum(len(b) for b in batches) == len(gen) + len(fix)
    for b in batches:
        assert len(b) * max(len(p.tgt) for p in b) <= 1500 or len(b) == 1
    batch = collate(batches[0], tok.pad_id, tok.bos_id)
    assert batch.tgt_in[:, 0].eq(tok.bos_id).all()
    assert (batch.tgt_in[:, 1:] == batch.tgt_out[:, :-1])[batch.tgt_out[:, :-1] != tok.pad_id].all()


def test_lr_schedule():
    assert lr_at(0, 0, 1.0, 10) == pytest.approx(0.1)
    assert lr_at(0.0, 10, 1.0, 10) == pytest.approx(1.0)
    assert lr_at(1.0, 500, 1.0, 10) == pytest.approx(0.1)


def test_validation_ppl_is_unsmoothed_over_non_pad_tokens(env):
    _, data, tok_path = env
    tok = load_tokenizer(tok_path)
    cfg = load_config("tiny")
    from mjava.model import ModelConfig, Transformer

    mc = dict(cfg["model"], vocab_size=tok.vocab_size)
    torch.manual_seed(0)
    model = Transformer(ModelConfig.from_dict(mc))
    gen, _ = load_pairs(tok, data, "val", 128, 256)
    r = evaluate_loss(model, gen, "cpu", max_tokens=700)  # several batches with different padding
    model.eval()
    tot, n = 0.0, 0
    for p in gen:  # one example at a time: no padding at all
        b = collate([p], tok.pad_id, tok.bos_id)
        with torch.no_grad():
            lg = model(b.src, b.tgt_in)
        tot += float(F.cross_entropy(lg[0], b.tgt_out[0], reduction="sum"))
        n += b.tgt_out.numel()
    assert r["tokens"] == n
    assert r["loss"] == pytest.approx(tot / n, rel=1e-4)
    assert r["ppl"] == pytest.approx(math.exp(tot / n), rel=1e-4)


@pytest.mark.slow
def test_tiny_overfits_and_checkpoint_loads(env):
    root, data, tok_path = env
    cfg = load_config("tiny")
    s = train(cfg, stage="seq2seq", data_dir=str(data), tokenizer_path=str(tok_path), name="ov",
              ckpt_root=str(root / "ck"), runs_root=str(root / "runs"), threads=4, log=_quiet)
    ck = root / "ck" / "ov"
    for f in ("model.pt", "config.yaml", "tokenizer.json"):
        assert (ck / f).exists()
    assert (root / "runs" / "ov" / "metrics.jsonl").exists() and (root / "runs" / "ov" / "metrics.csv").exists()
    assert s["best_val_ppl"] < 1.5, s

    gen = TransformerGenerator.from_checkpoint(ck)
    recs = read_jsonl(FIX / "train.jsonl")
    outs = gen.generate([build_generate_source(r["instruction"]) for r in recs], GenerationConfig(max_new_tokens=250))
    exact = sum(o[0] == r["java"] for o, r in zip(outs, recs)) / len(recs)
    assert exact > 0.95, exact
    # batched generation == one-at-a-time generation
    one = [gen.generate([build_generate_source(r["instruction"])])[0][0] for r in recs[:6]]
    assert one == [o[0] for o in outs[:6]]
    # sampling / beam / multi-return shapes
    cfgs = [GenerationConfig(max_new_tokens=250, num_beams=3, num_return_sequences=2),
            GenerationConfig(max_new_tokens=60, do_sample=True, temperature=0.8, top_k=10, num_return_sequences=3)]
    for c in cfgs:
        res = gen.generate([build_generate_source(r["instruction"]) for r in recs[:3]], c)
        assert len(res) == 3 and all(len(x) == c.num_return_sequences for x in res)
    beam = gen.generate([build_generate_source(recs[0]["instruction"])], GenerationConfig(max_new_tokens=250, num_beams=3))
    assert beam[0][0] == recs[0]["java"]


def test_resume_and_max_minutes(env):
    root, data, tok_path = env
    cfg = load_config("tiny", ["train.eval_every=10", "train.log_every=5"])
    kw = dict(stage="seq2seq", data_dir=str(data), tokenizer_path=str(tok_path), name="res",
              ckpt_root=str(root / "ck"), runs_root=str(root / "runs"), threads=2, log=_quiet)
    s1 = train(cfg, max_steps=20, **kw)
    assert s1["steps"] == 20
    cfg2 = load_config("tiny", ["train.eval_every=10", "train.log_every=5", "train.max_steps=40"])
    s2 = train(cfg2, resume=True, **kw)
    assert s2["steps"] == 40 and s2["stop_reason"] == "max_steps"
    lines = (root / "runs" / "res" / "metrics.jsonl").read_text().splitlines()
    assert len(lines) > 8
    # wall-clock budget: stops cleanly well before max_steps and still leaves a loadable best checkpoint
    cfg3 = load_config("tiny", ["train.max_steps=100000"])
    kw["name"] = "mm"
    s3 = train(cfg3, max_minutes=0.03, **kw)
    assert s3["stop_reason"] == "max_minutes" and s3["steps"] < 100000
    TransformerGenerator.from_checkpoint(root / "ck" / "mm")


def test_lm_stage_runs(env):
    root, data, tok_path = env
    cfg = load_config("tiny")
    s = train(cfg, stage="lm", data_dir=str(data), tokenizer_path=str(tok_path), name="lm",
              ckpt_root=str(root / "ck"), runs_root=str(root / "runs"), max_steps=10, threads=2, log=_quiet)
    assert s["steps"] == 10 and math.isfinite(s["best_val_loss"])
    # chain: seq2seq initialised from the lm checkpoint
    s2 = train(cfg, stage="seq2seq", data_dir=str(data), init_from=str(root / "ck" / "lm"), name="lm2s",
               ckpt_root=str(root / "ck"), runs_root=str(root / "runs"), max_steps=5, threads=2, log=_quiet)
    assert s2["steps"] == 5


def test_load_config_coerces_scientific_notation(tmp_path):
    cfg = load_config("tiny", ["train.lr=1e-3", "train.weight_decay=2E-4", "train.name=abc", "train.max_steps=7"])
    assert cfg["train"]["lr"] == 1e-3 and isinstance(cfg["train"]["lr"], float)
    assert cfg["train"]["weight_decay"] == 2e-4
    assert cfg["train"]["name"] == "abc" and cfg["train"]["max_steps"] == 7
    y = tmp_path / "c.yaml"
    y.write_text("train:\n  lr: 1e-3\n  betas: [0.9, 5e-1]\n", encoding="utf-8")
    c2 = load_config(y)
    assert c2["train"]["lr"] == 1e-3 and c2["train"]["betas"] == [0.9, 0.5]
    torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=c2["train"]["lr"])  # used to raise on a str lr
