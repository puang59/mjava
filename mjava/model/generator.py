"""``TransformerGenerator``: the ``CodeGenerator`` implementation used by the pipeline."""

from __future__ import annotations

from typing import Sequence

import torch

from mjava.interfaces import GenerationConfig
from mjava.model import decoding
from mjava.model.checkpoint import load_checkpoint
from mjava.model.transformer import Transformer
from mjava.tokenizer import JavaTokenizer


class TransformerGenerator:
    def __init__(self, model: Transformer, tokenizer: JavaTokenizer, device: str = "cpu", batch_size: int = 32):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size

    @classmethod
    def from_checkpoint(cls, ckpt_dir, device: str = "cpu", batch_size: int = 32) -> "TransformerGenerator":
        model, tok = load_checkpoint(ckpt_dir, device)
        return cls(model, tok, device, batch_size)

    def encode_sources(self, sources: Sequence[str]) -> list[list[int]]:
        ml = self.model.cfg.max_src_len
        return [ids[:ml] for ids in self.tokenizer.encode_batch(list(sources))]

    def _decode_ids(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special=True)

    @torch.inference_mode()
    def generate(self, sources: Sequence[str], config: GenerationConfig | None = None) -> list[list[str]]:
        cfg = config or GenerationConfig()
        n = max(1, cfg.num_return_sequences)
        ids = self.encode_sources(sources)
        order = sorted(range(len(ids)), key=lambda i: len(ids[i]))  # length bucketing
        results: list[list[str] | None] = [None] * len(ids)
        bs = max(1, self.batch_size // (cfg.num_beams if cfg.num_beams > 1 else (n if cfg.do_sample else 1)))
        was_training = self.model.training
        self.model.eval()
        for s in range(0, len(order), bs):
            idx = order[s : s + bs]
            src = decoding.pad_sequences([ids[i] for i in idx], self.model.cfg.pad_id, self.device)
            if cfg.num_beams > 1:
                res = decoding.beam_search(
                    self.model, src, cfg.num_beams, cfg.max_new_tokens, num_return=n
                )
                for i, hyps in zip(idx, res):
                    outs = [self._decode_ids(h) for h, _ in hyps]
                    while len(outs) < n:
                        outs.append(outs[-1] if outs else "")
                    results[i] = outs
            else:
                out = decoding.sample(
                    self.model, src, cfg.max_new_tokens, cfg.temperature, cfg.top_k,
                    greedy=not cfg.do_sample, num_samples=n if cfg.do_sample else 1,
                )
                seqs = [self._decode_ids(x) for x in decoding.strip_output(out, self.model.cfg.eos_id)]
                for j, i in enumerate(idx):
                    results[i] = seqs[j * n : (j + 1) * n] if cfg.do_sample else [seqs[j]] * n
        self.model.train(was_training)
        return results  # type: ignore[return-value]
