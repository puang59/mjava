"""RL unit tests: reward, spec lookup, loss gradient direction (tiny model, fake reward). No JDK needed."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))

from mjava.interfaces import CompileResult, Diagnostic  # noqa: E402
from mjava.model import ModelConfig, Transformer  # noqa: E402
from mjava.model.decoding import SampleOutput, sample, sequence_log_probs  # noqa: E402
from mjava.train.rl import (  # noqa: E402
    RL_DEFAULTS, RewardComputer, RewardInfo, SpecIndex, compute_advantages, policy_loss, reward_from_parts,
    rl_config, rl_step, RLItem,
)

SPEC = {
    "family": "data_class", "heldout": False, "imports": [],
    "types": [{"kind": "class", "name": "Student", "fields": [{"type": "String", "name": "name"}, {"type": "int", "name": "age"}],
               "methods": []}],
}
GOOD = "class Student {\n    String name;\n    int age;\n}\n"


class FakeCompiler:
    """Compiles iff braces balance and no 'BAD' marker; counts calls."""

    def __init__(self):
        self.calls = []

    def compile_many(self, sources):
        self.calls.append(list(sources))
        return [self.compile(s) for s in sources]

    def compile(self, s, filename="Main.java"):
        if "TIMEOUT" in s:
            return CompileResult(False, (Diagnostic("ERROR", -1, -1, "server.timeout", "compile server did not respond"),))
        if "BOOM" in s:
            raise RuntimeError("boom")
        ok = s.count("{") == s.count("}") and "BAD" not in s
        return CompileResult(ok, () if ok else (Diagnostic("ERROR", 1, 1, "x", "bad"),) * 2)

    def close(self):
        pass


def comp(rc_over=None):
    rc = dict(RL_DEFAULTS)
    rc.update(rc_over or {})
    return RewardComputer(FakeCompiler(), rc)


def test_reward_values():
    rc = comp()
    infos = rc.score([GOOD, "class A {}", "class Student {\n String name;", "", GOOD, "class Student { BAD }"],
                     [True, True, True, True, False, True], [SPEC] * 6)
    good, degenerate, broken, empty, truncated, bad = infos
    assert good.reward == pytest.approx(1.0) and good.compiled
    assert degenerate.compiled and degenerate.reward < 0.2  # compiles but misses the spec
    assert not broken.compiled and broken.reward == pytest.approx(0.3 * broken.conformance * 0.6)
    assert broken.reward < good.reward
    assert empty.reward == 0.0 and truncated.reward == 0.0  # truncated output (no eos) scores 0 even if it would compile
    assert bad.reward < good.reward and not bad.compiled


def test_reward_never_crashes_and_caches():
    rc = comp({"max_chars": 100})
    texts = [GOOD, GOOD, "TIMEOUT", "BOOM", "x" * 500, "�\x00 garbage"]
    infos = rc.score(texts, [True] * 6, [SPEC] * 6)
    assert [i.reward for i in infos][2:5] == [0.0, 0.0, 0.0]
    assert infos[2].failed
    assert len(rc.compiler.calls[0]) == 4  # GOOD compiled once; oversized never sent
    rc.score([GOOD], [True], [SPEC])
    assert rc.cache_hits >= 1 and GOOD not in [s for c in rc.compiler.calls[1:] for s in c]
    assert "TIMEOUT" not in rc.cache  # infrastructure failures are never cached


def test_no_spec_fallback_is_compile_only():
    rc = comp()
    a, b = rc.score([GOOD, "class A { BAD }"], [True, True], [None, {"kind": "class", "name": "Order"}])
    assert not a.has_spec and not b.has_spec
    assert a.reward == pytest.approx(0.5) and 0 < b.reward < 0.5


def test_reward_from_parts_penalty():
    assert reward_from_parts(False, 1.0, 5) == 0.0
    assert reward_from_parts(False, 1.0, 1) == pytest.approx(0.24)
    assert reward_from_parts(True, 0.7, 9) == 0.7


def test_advantages_shape_and_sign():
    adv = compute_advantages([1, 0, 0.5, 0.5], [0.5, 0.5], 2)
    assert adv.tolist() == [0.5, -0.5, 0.0, 0.0]


def test_rl_config_rejects_typos():
    with pytest.raises(ValueError):
        rl_config({"rl": {"learning_rate": 1}})
    assert rl_config({"rl": {"lr": 3e-4}})["lr"] == 3e-4


def test_spec_index_repair_via_id_and_java():
    gen = [{"id": "train-000001", "java": "class A {}", "spec": SPEC}, {"id": "train-000002", "java": "J2", "spec": {"kind": "x"}}]
    ix = SpecIndex(gen)
    assert ix.lookup({"id": "train-000001"}) == (SPEC, "id")
    assert ix.lookup({"id": "repair-train-000001", "java": "zzz"}) == (SPEC, "id")
    assert ix.lookup({"id": "weird", "java": "class A {}"}) == (SPEC, "java")
    assert ix.lookup({"id": "repair-train-000002", "java": "J2"})[0] is None  # spec without checkable elements


# ------------------------------------------------------------- gradient direction
def tiny_model(seed=0):
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=40, d_model=32, n_heads=2, n_enc_layers=1, n_dec_layers=1, d_ff=64, dropout=0.0,
                      max_src_len=16, max_tgt_len=12, pad_id=0, bos_id=1, eos_id=2)
    return Transformer(cfg)


def test_policy_loss_gradient_direction():
    model = tiny_model()
    src = torch.randint(4, 40, (2, 6))
    tokens = torch.tensor([[5, 6, 7, 2, 0], [8, 9, 10, 2, 0]])  # row 0 rewarded, row 1 penalised
    mask = tokens != 0
    out = SampleOutput(tokens, torch.zeros(2, 5), mask, torch.ones(2, dtype=torch.bool))
    adv = torch.tensor([1.0, -1.0])

    def seq_lp():
        with torch.no_grad():
            return sequence_log_probs(model, src, tokens, mask).sum(1)

    before = seq_lp()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(8):
        loss = policy_loss(model, src, out, adv, 1.0, 0)
        opt.zero_grad()
        loss.backward()
        opt.step()
    after = seq_lp()
    assert after[0] > before[0] + 0.1, (before, after)
    assert after[1] < before[1] - 0.1, (before, after)


def test_zero_advantage_gives_zero_gradient():
    model = tiny_model()
    src = torch.randint(4, 40, (2, 6))
    tokens = torch.tensor([[5, 6, 2], [8, 9, 2]])
    out = SampleOutput(tokens, torch.zeros(2, 3), tokens != 0, torch.ones(2, dtype=torch.bool))
    loss = policy_loss(model, src, out, torch.zeros(2), 1.0, 0)
    assert float(loss) == 0.0 and not loss.requires_grad


class SpyScorer:
    def __init__(self, fn):
        self.fn = fn
        self.seen = []

    def score(self, texts, finished, specs):
        self.seen.append((list(texts), list(finished)))
        return [RewardInfo(self.fn(t, f), compiled=True, conformance=1.0) for t, f in zip(texts, finished)]


class ToyTok:
    eos_id = 2

    def decode(self, ids, skip_special=True):
        return " ".join(map(str, ids))


def test_rl_step_with_fake_reward_improves_rewarded_behaviour():
    """Reward = fraction of tokens that equal 7 in the decoded sample (greedy gets its own reward, so the baseline moves).
    After some SCST steps (lambda_rl=1) the probability of token 7 under the policy must go up."""
    model = tiny_model(1)
    src = torch.randint(4, 40, (4, 6))
    items = [RLItem([int(x) for x in row], [5, 7, 7, 2], "gen", f"i{i}") for i, row in enumerate(src)]
    rc = dict(RL_DEFAULTS, lambda_rl=1.0, num_samples=4, temperature=1.0, top_k=0, mle_label_smoothing=0.0)
    scorer = SpyScorer(lambda t, f: sum(1 for x in t.split() if x == "7") / max(len(t.split()), 1) if f or True else 0.0)
    opt = torch.optim.SGD(model.parameters(), lr=0.3)

    def p7():
        with torch.no_grad():
            model.eval()
            from mjava.model.transformer import Masks  # noqa: F401
            tgt_in = torch.full((4, 1), 1, dtype=torch.long)
            lg = model(src, tgt_in)[:, -1]
            return torch.softmax(lg, -1)[:, 7].mean().item()

    before = p7()
    for _ in range(40):
        rl_step(model, ToyTok(), items, scorer, rc, "cpu", None, None, opt, clip=1.0)
    assert p7() > before, (before, p7())
