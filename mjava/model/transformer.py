"""Encoder-decoder Transformer in plain PyTorch (own attention / blocks).

Pre-LN blocks, learned positions, one token embedding shared by encoder input,
decoder input and (transposed) the output projection.

Mask convention everywhere: boolean, ``True`` = may attend / real token.

Incremental decoding::

    state = model.init_state(src, max_len=T)       # encodes once, allocates KV cache
    logits = model.decode_step(tok, state)         # tok: (B,) -> logits (B, V); advances state
    state.reorder(idx) / state.repeat_interleave(n)  # for beam search / multi-sample

``decode_step`` writes the cache in place and is inference-only (use it under
``torch.no_grad()``); training / RL re-scoring goes through ``forward``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 6000
    d_model: int = 256
    n_heads: int = 4
    n_enc_layers: int = 4
    n_dec_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    max_src_len: int = 192
    max_tgt_len: int = 384
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Masks:
    """Optional explicit masks for ``forward``: ``src_keep`` (B,S) and ``tgt_keep`` (B,T), True = real token."""

    src_keep: Optional[torch.Tensor] = None
    tgt_keep: Optional[torch.Tensor] = None


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dh = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.kv = nn.Linear(d_model, 2 * d_model)
        self.o = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def split(self, x: torch.Tensor) -> torch.Tensor:  # (B,T,D) -> (B,H,T,dh)
        B, T, _ = x.shape
        return x.view(B, T, self.h, self.dh).transpose(1, 2)

    def project_kv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        k, v = self.kv(x).chunk(2, dim=-1)
        return self.split(k), self.split(v)

    def attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        # softmax(q k^T / sqrt(dh)) v, via the fused torch kernel; mask: bool, True = keep
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0
        )
        B, H, T, dh = out.shape
        return self.o(out.transpose(1, 2).reshape(B, T, H * dh))

    def forward(
        self,
        x: torch.Tensor,
        mem: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        q = self.split(self.q(x))
        k, v = kv if kv is not None else self.project_kv(x if mem is None else mem)
        return self.attend(q, k, v, mask)

    def step_self(self, x: torch.Tensor, ck: torch.Tensor, cv: torch.Tensor, pos: int) -> torch.Tensor:
        """One-token self-attention against the cache buffers ck/cv (B,H,L,dh); writes slot ``pos``."""
        q = self.split(self.q(x))
        k, v = self.project_kv(x)
        ck[:, :, pos : pos + 1] = k
        cv[:, :, pos : pos + 1] = v
        return self.attend(q, ck[:, :, : pos + 1], cv[:, :, : pos + 1], None)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.drop(F.gelu(self.w1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.d_model)
        self.attn = MultiHeadAttention(c.d_model, c.n_heads, c.dropout)
        self.ln2 = nn.LayerNorm(c.d_model)
        self.ff = FeedForward(c.d_model, c.d_ff, c.dropout)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x), mask=mask))
        return x + self.drop(self.ff(self.ln2(x)))


class DecoderLayer(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.d_model)
        self.self_attn = MultiHeadAttention(c.d_model, c.n_heads, c.dropout)
        self.ln2 = nn.LayerNorm(c.d_model)
        self.cross_attn = MultiHeadAttention(c.d_model, c.n_heads, c.dropout)
        self.ln3 = nn.LayerNorm(c.d_model)
        self.ff = FeedForward(c.d_model, c.d_ff, c.dropout)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x, memory, self_mask, cross_mask):
        x = x + self.drop(self.self_attn(self.ln1(x), mask=self_mask))
        x = x + self.drop(self.cross_attn(self.ln2(x), mem=memory, mask=cross_mask))
        return x + self.drop(self.ff(self.ln3(x)))

    def step(self, x, ck, cv, pos, cross_kv, cross_mask):
        x = x + self.self_attn.step_self(self.ln1(x), ck, cv, pos)
        x = x + self.cross_attn(self.ln2(x), mask=cross_mask, kv=cross_kv)
        return x + self.ff(self.ln3(x))


@dataclass
class DecoderState:
    """KV cache for incremental decoding (all tensors batch-first)."""

    cross_k: list
    cross_v: list
    cross_mask: torch.Tensor  # (B,1,1,S) bool
    self_k: list  # per layer (B,H,L,dh) preallocated
    self_v: list
    pos: int = 0

    @property
    def batch_size(self) -> int:
        return self.cross_mask.size(0)

    def _apply(self, fn) -> None:
        self.cross_k = [fn(t) for t in self.cross_k]
        self.cross_v = [fn(t) for t in self.cross_v]
        self.cross_mask = fn(self.cross_mask)
        self.self_k = [fn(t) for t in self.self_k]
        self.self_v = [fn(t) for t in self.self_v]

    def reorder(self, idx: torch.Tensor) -> None:
        """Select/permute batch rows (beam search)."""
        self._apply(lambda t: t.index_select(0, idx))

    def repeat_interleave(self, n: int) -> None:
        self._apply(lambda t: t.repeat_interleave(n, dim=0))


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.embed = nn.Embedding(cfg.vocab_size, d, padding_idx=cfg.pad_id)
        self.src_pos = nn.Embedding(cfg.max_src_len, d)
        self.tgt_pos = nn.Embedding(cfg.max_tgt_len, d)
        self.drop = nn.Dropout(cfg.dropout)
        self.enc_layers = nn.ModuleList(EncoderLayer(cfg) for _ in range(cfg.n_enc_layers))
        self.dec_layers = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.n_dec_layers))
        self.enc_ln = nn.LayerNorm(d)
        self.dec_ln = nn.LayerNorm(d)
        self.scale = math.sqrt(d)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.embed.weight, std=self.cfg.d_model ** -0.5)
        with torch.no_grad():
            self.embed.weight[self.cfg.pad_id].zero_()
        nn.init.normal_(self.src_pos.weight, std=0.02)
        nn.init.normal_(self.tgt_pos.weight, std=0.02)
        # GPT-2 style residual scaling
        n = max(1, self.cfg.n_enc_layers + 2 * self.cfg.n_dec_layers)
        s = 0.02 / math.sqrt(2 * n)
        for layer in self.enc_layers:
            nn.init.normal_(layer.attn.o.weight, std=s)
            nn.init.normal_(layer.ff.w2.weight, std=s)
        for layer in self.dec_layers:
            nn.init.normal_(layer.self_attn.o.weight, std=s)
            nn.init.normal_(layer.cross_attn.o.weight, std=s)
            nn.init.normal_(layer.ff.w2.weight, std=s)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------ encode
    def _src_keep(self, src: torch.Tensor, masks: Optional[Masks]) -> torch.Tensor:
        if masks is not None and masks.src_keep is not None:
            return masks.src_keep
        return src != self.cfg.pad_id

    def encode(self, src: torch.Tensor, src_keep: Optional[torch.Tensor] = None):
        """src (B,S) -> (memory (B,S,D), cross_mask (B,1,1,S))."""
        if src_keep is None:
            src_keep = src != self.cfg.pad_id
        S = src.size(1)
        if S > self.cfg.max_src_len:
            raise ValueError(f"source length {S} > max_src_len {self.cfg.max_src_len}")
        pos = torch.arange(S, device=src.device)
        x = self.drop(self.embed(src) * self.scale + self.src_pos(pos))
        mask = src_keep[:, None, None, :]
        for layer in self.enc_layers:
            x = layer(x, mask)
        return self.enc_ln(x), mask

    # ------------------------------------------------------------------ forward
    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor, masks: Optional[Masks] = None) -> torch.Tensor:
        """Teacher-forced logits (B,T,V). tgt_in = [bos, y_0, ..., y_{T-2}]."""
        T = tgt_in.size(1)
        if T > self.cfg.max_tgt_len:
            raise ValueError(f"target length {T} > max_tgt_len {self.cfg.max_tgt_len}")
        memory, cross_mask = self.encode(src, self._src_keep(src, masks))
        tgt_keep = masks.tgt_keep if masks is not None and masks.tgt_keep is not None else tgt_in != self.cfg.pad_id
        causal = torch.ones(T, T, dtype=torch.bool, device=tgt_in.device).tril()
        self_mask = causal[None, None] & tgt_keep[:, None, None, :]
        pos = torch.arange(T, device=tgt_in.device)
        x = self.drop(self.embed(tgt_in) * self.scale + self.tgt_pos(pos))
        for layer in self.dec_layers:
            x = layer(x, memory, self_mask, cross_mask)
        x = self.dec_ln(x)
        return x @ self.embed.weight.t()

    # ------------------------------------------------------------- incremental
    @torch.no_grad()
    def init_state(
        self,
        src: torch.Tensor,
        max_len: int,
        src_keep: Optional[torch.Tensor] = None,
        num_repeat: int = 1,
    ) -> DecoderState:
        """Encode ``src`` once and allocate a KV cache for ``max_len`` decoder steps.
        ``num_repeat`` > 1 repeats every row that many times (beam / multi-sample)."""
        if max_len > self.cfg.max_tgt_len:
            raise ValueError(f"max_len {max_len} > max_tgt_len {self.cfg.max_tgt_len}")
        memory, cross_mask = self.encode(src, src_keep)
        ck, cv = [], []
        for layer in self.dec_layers:
            k, v = layer.cross_attn.project_kv(memory)
            ck.append(k)
            cv.append(v)
        B = src.size(0)
        h, dh = self.cfg.n_heads, self.cfg.d_model // self.cfg.n_heads
        sk = [torch.zeros(B, h, max_len, dh, device=src.device, dtype=memory.dtype) for _ in self.dec_layers]
        sv = [torch.zeros_like(t) for t in sk]
        st = DecoderState(ck, cv, cross_mask, sk, sv, 0)
        if num_repeat > 1:
            st.repeat_interleave(num_repeat)
        return st

    @torch.no_grad()
    def decode_step(self, tok: torch.Tensor, state: DecoderState) -> torch.Tensor:
        """Feed one token per row (B,) at position ``state.pos``; return logits (B,V)."""
        pos = state.pos
        if pos >= state.self_k[0].size(2):
            raise ValueError("KV cache is full")
        x = self.embed(tok)[:, None, :] * self.scale + self.tgt_pos.weight[pos]
        for i, layer in enumerate(self.dec_layers):
            x = layer.step(x, state.self_k[i], state.self_v[i], pos, (state.cross_k[i], state.cross_v[i]), state.cross_mask)
        state.pos = pos + 1
        x = self.dec_ln(x[:, 0])
        return x @ self.embed.weight.t()
