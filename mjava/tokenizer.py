"""Byte-level BPE tokenizer (HuggingFace ``tokenizers``) for instructions + Java.

* Byte-level => every string round-trips exactly (whitespace included).
* The ``SPECIAL_TOKENS`` of ``interfaces.py`` are atomic tokens with ids 0..7
  (in the order of ``SPECIAL_TOKENS``: pad=0, bos=1, eos=2, unk=3, gen=4, ...).

CLI::

    python -m mjava.tokenizer --data data --out checkpoints/tokenizer.json --vocab-size 6000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from mjava.interfaces import BOS, CODE, EOS, ERR, FIX, GEN, PAD, SPECIAL_TOKENS, UNK

CORPUS_SEP = "<|endoftext|>"
_TEXT_KEYS = ("instruction", "java", "broken_java", "error_text")


class JavaTokenizer:
    """Thin wrapper around a ``tokenizers.Tokenizer`` with special-id helpers."""

    def __init__(self, tok: Tokenizer):
        self.tok = tok
        self.pad_id = tok.token_to_id(PAD)
        self.bos_id = tok.token_to_id(BOS)
        self.eos_id = tok.token_to_id(EOS)
        self.unk_id = tok.token_to_id(UNK)
        self.gen_id = tok.token_to_id(GEN)
        self.fix_id = tok.token_to_id(FIX)
        self.err_id = tok.token_to_id(ERR)
        self.code_id = tok.token_to_id(CODE)

    @property
    def vocab_size(self) -> int:
        return self.tok.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text).ids

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        return [e.ids for e in self.tok.encode_batch(list(texts))]

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        return self.tok.decode(list(ids), skip_special_tokens=skip_special)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.tok.save(str(path))


def encode(tokenizer: JavaTokenizer, text: str) -> list[int]:
    return tokenizer.encode(text)


def decode(tokenizer: JavaTokenizer, ids: Sequence[int], skip_special: bool = True) -> str:
    return tokenizer.decode(ids, skip_special)


def load_tokenizer(path: str | Path) -> JavaTokenizer:
    return JavaTokenizer(Tokenizer.from_file(str(path)))


def _iter_texts(paths: Iterable[str | Path]) -> Iterator[str]:
    for p in paths:
        p = Path(p)
        if p.is_dir():
            files = sorted(list(p.glob("*.jsonl")) + list(p.glob("corpus_*.txt")))
        else:
            files = [p]
        for f in files:
            if f.suffix == ".jsonl":
                with open(f, encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        for k in _TEXT_KEYS:
                            if rec.get(k):
                                yield rec[k]
            else:
                text = f.read_text(encoding="utf-8")
                for chunk in text.split(CORPUS_SEP):
                    if chunk.strip():
                        yield chunk


def train_tokenizer(
    paths: Iterable[str | Path],
    out: str | Path,
    vocab_size: int = 6000,
    min_frequency: int = 2,
) -> JavaTokenizer:
    """Train a byte-level BPE on ``paths`` (``*.jsonl`` Example/RepairExample
    files, ``*.txt`` corpora, or directories containing them) and save to ``out``."""
    tok = Tokenizer(models.BPE(unk_token=UNK))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(_iter_texts(list(paths)), trainer=trainer)
    wrapped = JavaTokenizer(tok)
    wrapped.save(out)
    return wrapped


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Train the byte-level BPE tokenizer.")
    ap.add_argument("--data", nargs="+", default=["data"], help="files or directories (jsonl / corpus txt)")
    ap.add_argument("--out", default="checkpoints/tokenizer.json")
    ap.add_argument("--vocab-size", type=int, default=6000)
    ap.add_argument("--min-frequency", type=int, default=2)
    a = ap.parse_args(argv)
    t = train_tokenizer(a.data, a.out, a.vocab_size, a.min_frequency)
    print(f"saved {a.out} (vocab {t.vocab_size})")


if __name__ == "__main__":
    main()
