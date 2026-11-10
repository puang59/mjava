"""Evaluation tests: baseline, summaries, tiny end-to-end eval.json, compare, plots."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))

from mjava.eval.baseline import RetrievalBaseline, substitute, words  # noqa: E402
from mjava.eval.compare import main as compare_main, render_comparison  # noqa: E402
from mjava.eval.evaluate import run_eval, summarize  # noqa: E402
from mjava.eval.plots import main as plots_main  # noqa: E402


def _quiet(_m):
    pass


# ----------------------------------------------------------------------- pure unit tests
def test_substitute_renames_identifiers_and_camel_parts():
    java = "class Student {\n    String name;\n    int age;\n    String getName() { return name; }\n}\n"
    out = substitute(java, words("Create a class Student with fields name as String and age as int."),
                     words("Create a class Order with fields title as String and age as int."))
    assert "class Order" in out and "String title;" in out and "getTitle()" in out and "return title;" in out
    assert "Student" not in out and "age" in out


def test_retrieval_picks_closest_instruction():
    recs = [
        {"instruction": "Create a class Student with fields name as String", "java": "class Student { String name; }"},
        {"instruction": "Write an enum Color with constants RED, GREEN", "java": "enum Color { RED, GREEN }"},
    ]
    b = RetrievalBaseline(recs, max_df=1.0)
    assert b.nearest("define enum Planet constants MARS, VENUS") == 1
    out = b.generate_one("Create a class Order with fields title as String")
    assert out == "class Order { String title; }"


def _rec(ok0, conf0, succ, conf_f, attempts, fam="a", held=False):
    return {"family": fam, "heldout": held, "a0_ok": ok0, "a0_exact": False, "a0_conf": conf0, "final_ok": succ is not None,
            "final_conf": conf_f, "final_exact": False, "succeeded_at": succ, "attempts": attempts}


def test_summarize_numbers():
    recs = [_rec(True, 1.0, 0, 1.0, 1), _rec(False, 0.5, 2, 1.0, 3), _rec(False, 0.5, 1, 0.5, 2), _rec(False, 0.0, None, 0.0, 4)]
    s = summarize(recs, 3)
    assert s["first_attempt"]["compile_rate"] == 0.25 and s["first_attempt"]["full_rate"] == 0.25
    assert [s["correction"]["success_at"][str(k)] for k in range(4)] == [0.25, 0.5, 0.75, 0.75]
    assert s["correction"]["full_success_at"]["3"] == 0.5  # one converged-but-wrong
    assert s["correction"]["converged_nonconformant_rate"] == 0.25
    assert s["correction"]["mean_attempts"] == 2.5


# --------------------------------------------------------------- end to end with a tiny model
pytestmark_e2e = [pytest.mark.slow, pytest.mark.javac]


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    from mjava.data.build import build
    from mjava.tokenizer import train_tokenizer
    from mjava.train.common import load_config
    from mjava.train.supervised import train

    root = tmp_path_factory.mktemp("ev")
    data = root / "data"
    build(data, 120, 24, 30, 30, 10, 12, seed=5, workers=1, log=_quiet)
    tok_path = root / "tokenizer.json"
    train_tokenizer([data], tok_path, vocab_size=600)
    cfg = load_config("tiny", ["train.max_steps=25", "train.eval_every=25", "train.val_examples=24", "train.log_every=5"])
    train(cfg, stage="seq2seq", data_dir=str(data), tokenizer_path=str(tok_path), name="sup",
          ckpt_root=str(root / "ck"), runs_root=str(root / "runs"), threads=2, log=_quiet)
    return root, data


def check_group(g, max_retries):
    for k in ("compile_rate", "exact_match", "mean_conformance", "full_rate"):
        assert 0 <= g["first_attempt"][k] <= 1
    sa = [g["correction"]["success_at"][str(k)] for k in range(max_retries + 1)]
    fa = [g["correction"]["full_success_at"][str(k)] for k in range(max_retries + 1)]
    assert sa == sorted(sa) and fa == sorted(fa) and all(0 <= x <= 1 for x in sa + fa)
    assert all(f <= s + 1e-9 for f, s in zip(fa, sa))
    assert sa[0] == pytest.approx(g["first_attempt"]["compile_rate"])
    assert 1 <= g["correction"]["mean_attempts"] <= max_retries + 1
    assert g["first_attempt"]["full_rate"] <= g["first_attempt"]["compile_rate"] + 1e-9


@pytest.mark.slow
@pytest.mark.javac
def test_eval_transformer_and_baseline_consistent(env):
    root, data = env
    kw = dict(data_dir=str(data), split="test", runs_root=str(root / "runs"), log=_quiet)
    r = run_eval(name="m", checkpoint=str(root / "ck" / "sup"), max_retries=2, limit=16, **kw)
    j = json.loads((root / "runs" / "m" / "eval.json").read_text())
    assert j == json.loads(json.dumps(r))
    assert j["n"] == 16 and j["system"] == "transformer" and j["perplexity"]["ppl"] > 1
    check_group(j, 2)
    assert sum(g["n"] for g in j["by_family"].values()) == j["n"]
    for g in j["by_family"].values():
        check_group(g, 2)
    assert sum(g["n"] for g in j["heldout"].values()) == j["n"]
    assert 0 <= j["repair"]["compile_rate"] <= 1 and j["repair"]["n"] == 12
    ex = [json.loads(x) for x in (root / "runs" / "m" / "eval_examples.jsonl").read_text().splitlines()]
    assert len(ex) == 16
    assert (root / "runs" / "m" / "eval.md").read_text().startswith("# Evaluation: m")

    b = run_eval(name="b", baseline="retrieval", max_retries=2, limit=16, **kw)
    check_group(b, 2)
    assert b["system"] == "retrieval_baseline" and b["perplexity"] is None and b["repair"] is None
    assert b["correction"]["success_at"]["0"] == b["correction"]["success_at"]["2"]

    md = render_comparison([b, j], ["retrieval", "supervised"])
    assert "| retrieval |" in md and "| supervised |" in md and "success@2" in md and "By task family" in md
    out = root / "cmp.md"
    compare_main([str(root / "runs" / "b"), str(root / "runs" / "m" / "eval.json"), "--labels", "a,b", "--out", str(out)])
    assert out.read_text().count("\n") > 5

    with pytest.raises(SystemExit):
        run_eval(name="x", checkpoint=str(root / "nope"), **kw)


@pytest.mark.slow
@pytest.mark.javac
def test_plots_from_supervised_and_rl_metrics(env):
    from mjava.train.common import load_config
    from mjava.train.rl import train_rl

    root, data = env
    cfg = load_config("tiny", ["rl.lr=1e-3", "rl.max_steps=4", "rl.eval_every=2", "rl.log_every=1", "rl.val_gen=6",
                               "rl.val_fix=3", "rl.batch_examples=6", "rl.warmup_steps=1"])
    train_rl(cfg, data_dir=str(data), init_from=str(root / "ck" / "sup"), name="rlp", ckpt_root=str(root / "ck"),
             runs_root=str(root / "runs"), threads=2, log=_quiet)
    out = root / "plots"
    plots_main(["--run", str(root / "runs" / "sup"), "--run", str(root / "runs" / "rlp"), "--out", str(out)])
    names = {p.name for p in out.glob("*.png")}
    assert {"loss.png", "perplexity.png", "rl_reward.png", "rl_greedy_compile.png", "rl_validation.png"} <= names
    assert all((out / n).stat().st_size > 2000 for n in names)
    with pytest.raises(SystemExit):
        plots_main(["--run", str(root / "missing")])
