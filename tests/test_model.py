import torch

from mjava.model import ModelConfig, Transformer, beam_search, pad_sequences, sample, sequence_log_probs, strip_output
from mjava.model.transformer import Masks

PAD, BOS, EOS = 0, 1, 2


def tiny_model(seed=0, vocab=50, **kw):
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=vocab, d_model=32, n_heads=4, n_enc_layers=2, n_dec_layers=2, d_ff=64,
                      dropout=0.0, max_src_len=24, max_tgt_len=40, **kw)
    m = Transformer(cfg)
    # larger weights so the random model is not near-uniform / degenerate
    with torch.no_grad():
        for p in m.parameters():
            if p.ndim >= 2:
                p.mul_(8.0)
    return m.eval()


def rand_src(B=3, S=10, lens=(10, 6, 3), vocab=50, seed=1):
    g = torch.Generator().manual_seed(seed)
    src = torch.randint(4, vocab, (B, S), generator=g)
    for i, n in enumerate(lens):
        src[i, n:] = PAD
    return src


def test_no_future_leakage():
    m = tiny_model()
    src = rand_src()
    tgt = torch.randint(4, 50, (3, 12))
    base = m(src, tgt)
    for t in (0, 3, 7):
        tgt2 = tgt.clone()
        tgt2[:, t + 1 :] = torch.randint(4, 50, tgt2[:, t + 1 :].shape)
        out = m(src, tgt2)
        assert torch.allclose(base[:, : t + 1], out[:, : t + 1], atol=1e-5)
        assert not torch.allclose(base[:, t + 1 :], out[:, t + 1 :], atol=1e-5)


def test_padding_does_not_change_non_pad_outputs():
    m = tiny_model()
    src = rand_src()
    keep = src != PAD
    tgt = torch.randint(4, 50, (3, 12))
    tkeep = torch.ones(3, 12, dtype=torch.bool)
    tkeep[1, 8:] = False
    tkeep[2, 5:] = False
    masks = Masks(keep, tkeep)
    base = m(src, tgt, masks)
    src2, tgt2 = src.clone(), tgt.clone()
    src2[~keep] = torch.randint(4, 50, (int((~keep).sum()),))
    tgt2[~tkeep] = torch.randint(4, 50, (int((~tkeep).sum()),))
    out = m(src2, tgt2, masks)
    assert torch.allclose(base[tkeep], out[tkeep], atol=1e-5)
    # default (pad-id derived) masks: batched padded == unpadded single example
    solo = m(src[2:3, :3], tgt[2:3, :5])
    padded = m(src, tgt)[2, :5]
    assert torch.allclose(solo[0], padded, atol=1e-5)


def test_cached_decoding_matches_full_forward():
    m = tiny_model()
    src = rand_src()
    tgt_in = torch.cat([torch.full((3, 1), BOS), torch.randint(4, 50, (3, 14))], 1)
    full = m(src, tgt_in)
    state = m.init_state(src, max_len=15)
    for t in range(15):
        step = m.decode_step(tgt_in[:, t], state)
        assert torch.allclose(step, full[:, t], atol=1e-4), t


def test_batched_greedy_equals_per_example():
    m = tiny_model()
    src = rand_src(B=4, lens=(10, 7, 5, 3))
    out = sample(m, src, max_new_tokens=20, greedy=True)
    batched = strip_output(out, EOS)
    for i in range(4):
        row = src[i : i + 1, : int((src[i] != PAD).sum())]
        single = strip_output(sample(m, row, max_new_tokens=20, greedy=True), EOS)[0]
        assert single == batched[i]


def test_greedy_equals_beam1_and_beam_scores_sorted():
    m = tiny_model(seed=3)
    src = rand_src(B=3)
    greedy = strip_output(sample(m, src, max_new_tokens=18, greedy=True), EOS)
    beam1 = beam_search(m, src, num_beams=1, max_new_tokens=18)
    assert [b[0][0] for b in beam1] == greedy
    beam4 = beam_search(m, src, num_beams=4, max_new_tokens=18, num_return=3)
    for hyps in beam4:
        assert len(hyps) == 3
        scores = [s for _, s in hyps]
        assert scores == sorted(scores, reverse=True)


def test_sample_log_probs_mask_and_grad():
    m = tiny_model(seed=5)
    src = rand_src()
    # force early, row-dependent <eos> so sequences finish at different steps
    orig = m.decode_step

    def patched(tok, state):
        lg = orig(tok, state)
        if state.pos > 0:
            lg = lg.clone()
            lg[:, EOS] += 1e6 * (tok % 4 == 0).float()
        return lg

    m.decode_step = patched
    g = torch.Generator().manual_seed(0)
    out = sample(m, src, max_new_tokens=25, temperature=6.0, num_samples=4, generator=g)
    assert out.tokens.shape[0] == 12
    lens = out.lengths
    assert len(set(lens.tolist())) > 1, "test needs sequences finishing at different steps"
    for i in range(12):
        n = int(lens[i])
        if out.finished[i]:
            assert out.tokens[i, n - 1] == EOS
            assert (out.tokens[i, n:] == PAD).all()
        assert out.mask[i, :n].all() and not out.mask[i, n:].any()
        assert (out.log_probs[i, n:] == 0).all()
    m.decode_step = orig

    # with_grad: values equal the no-grad ones, gradients flow, masked positions get none
    m.eval()
    g1 = torch.Generator().manual_seed(7)
    a = sample(m, src, max_new_tokens=12, num_samples=2, generator=g1)
    g2 = torch.Generator().manual_seed(7)
    b = sample(m, src, max_new_tokens=12, num_samples=2, generator=g2, with_grad=True)
    assert torch.equal(a.tokens, b.tokens)
    assert torch.allclose(a.log_probs, b.log_probs.detach(), atol=1e-4)
    assert b.log_probs.requires_grad and not a.log_probs.requires_grad
    (b.log_probs * b.mask).sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())
    assert (b.log_probs <= 1e-6).all()


def test_top_k_and_temperature_logprobs_consistent():
    m = tiny_model(seed=2)
    src = rand_src()
    g = torch.Generator().manual_seed(1)
    out = sample(m, src, max_new_tokens=10, temperature=0.7, top_k=5, generator=g)
    re = sequence_log_probs(m, src, out.tokens, out.mask, temperature=0.7, top_k=5)
    assert torch.allclose(out.log_probs, re, atol=1e-4)


def test_pad_sequences():
    t = pad_sequences([[1, 2, 3], [4]], 0)
    assert t.tolist() == [[1, 2, 3], [4, 0, 0]]
