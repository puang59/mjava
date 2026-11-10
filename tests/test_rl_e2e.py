"""Tiny end-to-end RL run with the real compile server (seconds)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))

from mjava.data.build import build  # noqa: E402
from mjava.model import TransformerGenerator  # noqa: E402
from mjava.tokenizer import load_tokenizer, train_tokenizer  # noqa: E402
from mjava.train.common import load_config, read_jsonl  # noqa: E402
from mjava.train.rl import SpecIndex, load_items, train_rl  # noqa: E402
from mjava.train.supervised import train  # noqa: E402

pytestmark = [pytest.mark.slow, pytest.mark.javac]


def _quiet(_m):
    pass


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("rl")
    data = root / "data"
    build(data, 120, 24, 24, 40, 12, 12, seed=3, workers=1, log=_quiet)
    tok_path = root / "tokenizer.json"
    train_tokenizer([data], tok_path, vocab_size=600)
    cfg = load_config("tiny", ["train.max_steps=25", "train.eval_every=25", "train.val_examples=24"])
    train(cfg, stage="seq2seq", data_dir=str(data), tokenizer_path=str(tok_path), name="sup",
          ckpt_root=str(root / "ck"), runs_root=str(root / "runs"), threads=2, log=_quiet)
    return root, data


def test_specs_found_for_generate_and_repair_examples(env):
    root, data = env
    tok = load_tokenizer(root / "tokenizer.json")
    index = SpecIndex.from_data(data)
    gen, fix = load_items(tok, data, "train", 256, 512, index)
    assert len(gen) > 50 and len(fix) > 10
    assert all(it.spec is not None and it.spec_via == "id" for it in gen)
    assert all(it.spec is not None for it in fix), sum(it.spec is None for it in fix)
    # the spec found for a repair example describes its reference java
    from mjava.data.conformance import spec_conformance
    recs = {r["id"]: r for r in read_jsonl(data / "repair_train.jsonl")}
    for it in fix[:10]:
        assert spec_conformance(it.spec, recs[it.id]["java"]) == 1.0


def test_rl_run_checkpoint_metrics_resume(env):
    root, data = env
    over = ["rl.lr=1e-3", "rl.max_steps=6", "rl.eval_every=3", "rl.log_every=2", "rl.val_gen=8", "rl.val_fix=4",
            "rl.batch_examples=8", "rl.num_samples=2", "rl.warmup_steps=2"]
    cfg = load_config("tiny", over)
    kw = dict(data_dir=str(data), init_from=str(root / "ck" / "sup"), name="rl", ckpt_root=str(root / "ck"),
              runs_root=str(root / "runs"), threads=2, log=_quiet)
    s = train_rl(cfg, **kw)
    assert s["steps"] == 6 and s["baseline"] is not None
    ck = root / "ck" / "rl"
    for f in ("model.pt", "config.yaml", "tokenizer.json", "last.pt"):
        assert (ck / f).exists()
    TransformerGenerator.from_checkpoint(ck)  # loadable by the pipeline
    rows = [json.loads(x) for x in (root / "runs" / "rl" / "metrics.jsonl").read_text().splitlines()]
    val = [r for r in rows if r["split"] == "val"]
    tr = [r for r in rows if r["split"] == "train"]
    assert val[0]["step"] == 0 and len(val) >= 3 and len(tr) == 3  # baseline + steps 3, 6 + final
    for r in val:
        assert 0 <= r["compile_rate"] <= 1 and 0 <= r["full_rate"] <= r["compile_rate"] + 1e-9 and r["ppl"] > 0
    for r in tr:
        assert 0 <= r["adv_nonzero_frac"] <= 1 and "reward" in r and "greedy_compile" in r
    # best checkpoint is never worse than the supervised start
    assert s["best"]["full_rate"] >= s["baseline"]["full_rate"]

    # resume continues the step counter and appends to the same log
    cfg2 = load_config("tiny", over[:1] + ["rl.max_steps=9"] + over[2:])
    s2 = train_rl(cfg2, resume=True, **kw)
    assert s2["steps"] == 9
    rows2 = (root / "runs" / "rl" / "metrics.jsonl").read_text().splitlines()
    assert len(rows2) > len(rows)

    # wall-clock budget: stops cleanly and leaves a loadable checkpoint
    cfg3 = load_config("tiny", over[:1] + ["rl.max_steps=100000"] + over[2:])
    kw["name"] = "rl_mm"
    s3 = train_rl(cfg3, max_minutes=0.02, **kw)
    assert s3["stop_reason"] == "max_minutes" and s3["steps"] < 100000
    TransformerGenerator.from_checkpoint(root / "ck" / "rl_mm")


def test_startup_errors_are_actionable(env, tmp_path):
    root, data = env
    cfg = load_config("tiny")
    with pytest.raises(SystemExit, match="init-from"):
        train_rl(cfg, data_dir=str(data), name="x", ckpt_root=str(tmp_path), runs_root=str(tmp_path), log=_quiet)
    with pytest.raises(SystemExit, match="model.pt"):
        train_rl(cfg, data_dir=str(data), init_from=str(tmp_path / "nope"), name="x", ckpt_root=str(tmp_path),
                 runs_root=str(tmp_path), log=_quiet)
    with pytest.raises(SystemExit, match="train.jsonl"):
        train_rl(cfg, data_dir=str(tmp_path / "nodata"), init_from=str(root / "ck" / "sup"), name="x",
                 ckpt_root=str(tmp_path), runs_root=str(tmp_path), log=_quiet)
    with pytest.raises(SystemExit, match="differ"):
        train_rl(cfg, data_dir=str(data), init_from=str(root / "ck" / "sup"), name="sup", ckpt_root=str(root / "ck"),
                 runs_root=str(tmp_path), log=_quiet)
