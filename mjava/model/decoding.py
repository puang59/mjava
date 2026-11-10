"""Batched greedy / sampling / beam-search decoding with a KV cache.

RL-facing API (``sample``)
--------------------------
``sample(model, src, max_new_tokens=..., temperature=1.0, top_k=0, greedy=False,
with_grad=False, num_samples=1, generator=None) -> SampleOutput``

* ``src``: LongTensor (B, S) of padded encoder sources (pad id = ``model.cfg.pad_id``).
  With ``num_samples=n`` every source is decoded n times; output row ``b*n + j``
  is sample j of source b.
* The returned ``SampleOutput`` has, for N = B*num_samples rows and T steps:
    - ``tokens``   (N, T) long: generated ids *excluding* the leading ``<bos>``;
      the ``<eos>`` is included when produced; everything after it is ``pad_id``.
    - ``mask``     (N, T) bool: True for positions that count (up to and including
      ``<eos>``). Positions after ``<eos>`` are False.
    - ``log_probs``(N, T) float: log p(token_t | prefix, src) under the sampling
      distribution (logits / temperature, top-k filtered, ``<pad>``/``<bos>``
      forbidden), 0.0 where ``mask`` is False. With ``with_grad=True`` these carry
      gradients to the model parameters (they are recomputed by one teacher-forced
      forward pass over the sampled tokens, in the model's *current* train/eval
      mode; the sampling pass itself always runs in eval mode, no grad, with the
      KV cache). With ``with_grad=False`` they are detached values from the fast path.
    - ``finished`` (N,) bool: an ``<eos>`` was produced (False = truncated).
    - ``lengths``  (N,) long: ``mask.sum(1)``.
  Sequence log-prob: ``(out.log_probs * out.mask).sum(1)``; length-normalised:
  divide by ``out.lengths``.  Decoded ids for row i: ``out.tokens[i][out.mask[i]]``
  (drop the trailing eos if ``finished[i]``).
* ``sequence_log_probs(model, src, tokens, mask, ...)`` recomputes the same
  log-probs for arbitrary (e.g. previously sampled / greedy) tokens with grad.
* ``greedy=True`` is the argmax decode (the self-critical baseline); it returns
  the same structure.  Dropout in the rescoring pass is the caller's choice: call
  ``model.eval()`` for exact on-policy log-probs, or ``model.train()`` to match a
  training-mode MLE loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from mjava.model.transformer import Masks, Transformer

NEG_INF = float("-inf")


@dataclass
class SampleOutput:
    tokens: torch.Tensor
    log_probs: torch.Tensor
    mask: torch.Tensor
    finished: torch.Tensor

    @property
    def lengths(self) -> torch.Tensor:
        return self.mask.sum(1)


def pad_sequences(seqs: Sequence[Sequence[int]], pad_id: int, device=None) -> torch.Tensor:
    L = max((len(s) for s in seqs), default=0)
    out = torch.full((len(seqs), max(L, 1)), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        if len(s):
            out[i, : len(s)] = torch.tensor(list(s), dtype=torch.long)
    return out.to(device) if device is not None else out


def process_logits(logits: torch.Tensor, model: Transformer, temperature: float = 1.0, top_k: int = 0) -> torch.Tensor:
    """Sampling distribution logits: forbid pad/bos, temperature, top-k."""
    forbid = torch.tensor([model.cfg.pad_id, model.cfg.bos_id], device=logits.device)
    logits = logits.index_fill(-1, forbid, NEG_INF)  # out of place (autograd safe)
    if temperature != 1.0:
        logits = logits / temperature
    if top_k and top_k < logits.size(-1):
        kth = logits.topk(top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, NEG_INF)
    return logits


def sequence_log_probs(
    model: Transformer,
    src: torch.Tensor,
    tokens: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    src_keep: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-token log-probs (N,T) of ``tokens`` (as returned by ``sample``) with gradients,
    0 where ``mask`` is False. ``src`` must have N rows (repeat_interleave for num_samples)."""
    N, T = tokens.shape
    bos = torch.full((N, 1), model.cfg.bos_id, dtype=torch.long, device=tokens.device)
    tgt_in = torch.cat([bos, tokens[:, :-1]], dim=1)
    tgt_keep = torch.cat([torch.ones(N, 1, dtype=torch.bool, device=tokens.device), mask[:, :-1]], dim=1)
    logits = model(src, tgt_in, Masks(src_keep=src_keep, tgt_keep=tgt_keep))
    logits = process_logits(logits, model, temperature, top_k)
    logp = F.log_softmax(logits.float(), dim=-1)
    lp = logp.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    return torch.where(mask, lp, torch.zeros_like(lp))


def sample(
    model: Transformer,
    src: torch.Tensor,
    max_new_tokens: int = 384,
    temperature: float = 1.0,
    top_k: int = 0,
    greedy: bool = False,
    with_grad: bool = False,
    num_samples: int = 1,
    generator: Optional[torch.Generator] = None,
) -> SampleOutput:
    """Batched incremental decode. See the module docstring for the full contract."""
    cfg = model.cfg
    T = min(max_new_tokens, cfg.max_tgt_len)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            state = model.init_state(src, T, num_repeat=num_samples)
            N = state.batch_size
            dev = src.device
            cur = torch.full((N,), cfg.bos_id, dtype=torch.long, device=dev)
            done = torch.zeros(N, dtype=torch.bool, device=dev)
            toks, lps, masks = [], [], []
            for _ in range(T):
                logits = model.decode_step(cur, state).float()
                logits = process_logits(logits, model, temperature if not greedy else 1.0, top_k if not greedy else 0)
                logp = F.log_softmax(logits, dim=-1)
                if greedy:
                    nxt = logits.argmax(-1)
                else:
                    nxt = torch.multinomial(logp.exp(), 1, generator=generator).squeeze(1)
                lp = logp.gather(1, nxt[:, None]).squeeze(1)
                active = ~done
                nxt = torch.where(active, nxt, torch.full_like(nxt, cfg.pad_id))
                lp = torch.where(active, lp, torch.zeros_like(lp))
                toks.append(nxt)
                lps.append(lp)
                masks.append(active)
                done = done | (nxt == cfg.eos_id)
                cur = nxt
                if bool(done.all()):
                    break
            tokens = torch.stack(toks, 1)
            mask = torch.stack(masks, 1)
            log_probs = torch.stack(lps, 1)
            finished = (tokens == cfg.eos_id).any(1)
    finally:
        model.train(was_training)
    if with_grad:
        s = src.repeat_interleave(num_samples, 0) if num_samples > 1 else src
        log_probs = sequence_log_probs(
            model, s, tokens, mask, temperature if not greedy else 1.0, top_k if not greedy else 0
        )
    return SampleOutput(tokens, log_probs, mask, finished)


def greedy_decode(model: Transformer, src: torch.Tensor, max_new_tokens: int = 384) -> SampleOutput:
    return sample(model, src, max_new_tokens, greedy=True)


def strip_output(out: SampleOutput, eos_id: int) -> list[list[int]]:
    """SampleOutput -> list of id lists without eos/pad."""
    res = []
    for row, m in zip(out.tokens.tolist(), out.mask.tolist()):
        ids = [t for t, k in zip(row, m) if k and t != eos_id]
        res.append(ids)
    return res


@torch.no_grad()
def beam_search(
    model: Transformer,
    src: torch.Tensor,
    num_beams: int = 4,
    max_new_tokens: int = 384,
    num_return: int = 1,
    length_penalty: float = 1.0,
) -> list[list[tuple[list[int], float]]]:
    """Batched beam search with KV-cache reordering.

    Returns, per source, up to ``num_return`` ``(token_ids_without_eos, score)`` best first;
    score = sum log p / length**length_penalty. ``num_beams=1`` equals greedy decoding.
    A hypothesis list stops growing once ``num_beams`` finished hypotheses exist."""
    cfg = model.cfg
    K = num_beams
    num_return = min(num_return, K)
    T = min(max_new_tokens, cfg.max_tgt_len)
    was_training = model.training
    model.eval()
    try:
        B = src.size(0)
        dev = src.device
        state = model.init_state(src, T, num_repeat=K)
        scores = torch.zeros(B, K, device=dev)
        scores[:, 1:] = NEG_INF
        seqs = torch.zeros(B * K, 0, dtype=torch.long, device=dev)
        cur = torch.full((B * K,), cfg.bos_id, dtype=torch.long, device=dev)
        finished: list[list[tuple[float, list[int]]]] = [[] for _ in range(B)]
        done = [False] * B
        V = cfg.vocab_size
        ar = torch.arange(K)
        for t in range(T):
            logits = process_logits(model.decode_step(cur, state).float(), model)
            logp = F.log_softmax(logits, dim=-1)
            cand = (scores.view(B * K, 1) + logp).view(B, K * V)
            top_s, top_i = cand.topk(min(2 * K, K * V), dim=1)
            top_s_l, top_i_l = top_s.tolist(), top_i.tolist()
            new_beam = torch.zeros(B, K, dtype=torch.long)
            new_tok = torch.full((B, K), cfg.pad_id, dtype=torch.long)
            new_scores = torch.full((B, K), NEG_INF)
            for b in range(B):
                if done[b]:
                    new_beam[b] = ar
                    continue
                n_alive = 0
                for r, (s, ix) in enumerate(zip(top_s_l[b], top_i_l[b])):
                    if s == NEG_INF:
                        break
                    beam, tk = divmod(ix, V)
                    if tk == cfg.eos_id:
                        if r < K:
                            hyp = seqs[b * K + beam].tolist()
                            finished[b].append((s / ((len(hyp) + 1) ** length_penalty), hyp))
                    elif n_alive < K:
                        new_beam[b, n_alive] = beam
                        new_tok[b, n_alive] = tk
                        new_scores[b, n_alive] = s
                        n_alive += 1
                    if n_alive == K and r >= K - 1:
                        break
                if len(finished[b]) >= K or n_alive == 0:
                    done[b] = True
                    new_beam[b] = ar
                    new_scores[b] = NEG_INF
                    new_tok[b] = cfg.pad_id
                else:
                    for j in range(n_alive, K):  # fewer than K live candidates
                        new_beam[b, j] = new_beam[b, 0]
            if all(done):
                break
            flat = (new_beam.to(dev) + (torch.arange(B, device=dev) * K)[:, None]).view(-1)
            state.reorder(flat)
            seqs = torch.cat([seqs[flat], new_tok.to(dev).view(-1, 1)], dim=1)
            scores = new_scores.to(dev)
            cur = new_tok.to(dev).view(-1)
        # add still-alive (truncated) beams
        seq_l = seqs.tolist()
        sc_l = scores.tolist()
        results = []
        for b in range(B):
            hyps = list(finished[b])
            if not done[b]:
                for k in range(K):
                    if sc_l[b][k] != NEG_INF:
                        h = seq_l[b * K + k]
                        hyps.append((sc_l[b][k] / (max(len(h), 1) ** length_penalty), h))
            hyps.sort(key=lambda x: -x[0])
            results.append([(h, s) for s, h in hyps[:num_return]])
        return results
    finally:
        model.train(was_training)
