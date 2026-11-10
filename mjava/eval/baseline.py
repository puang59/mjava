"""Non-neural baseline: nearest training instruction (TF-IDF cosine) + identifier substitution.

For a test instruction the training example with the most similar instruction is retrieved; the instruction tokens
of both are aligned (difflib) and every differing 1:1 token pair (``Student -> Order``, ``name -> title``, ``int ->
double``, ``5 -> 10``) is applied to the retrieved Java, also inside camel-case identifiers (``getName -> getTitle``).
This is the "template-based generation" reference point: it copes with rephrased names but not with a different
structure, so it should fail exactly where the Transformer is supposed to generalise.
"""
from __future__ import annotations

import difflib
import math
import re
from collections import Counter
from typing import Sequence

import numpy as np

WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
HUMP = re.compile(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])")


def words(text: str) -> list[str]:
    return WORD.findall(text)


def _variants(a: str, b: str) -> dict[str, str]:
    out = {a: b}
    if a[:1].isalpha() and b[:1].isalpha():
        out[a[:1].upper() + a[1:]] = b[:1].upper() + b[1:]
        out[a[:1].lower() + a[1:]] = b[:1].lower() + b[1:]
        out[a.upper()] = b.upper()
    return out


def substitute(java: str, src_words: Sequence[str], dst_words: Sequence[str]) -> str:
    """Apply the token alignment ``src_words -> dst_words`` to ``java``."""
    sm = difflib.SequenceMatcher(None, list(src_words), list(dst_words), autojunk=False)
    mapping: dict[str, str] = {}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace" and i2 - i1 == j2 - j1:
            for a, b in zip(src_words[i1:i2], dst_words[j1:j2]):
                if a != b:
                    for k, v in _variants(a, b).items():
                        mapping.setdefault(k, v)
    if not mapping:
        return java

    def fix(m: re.Match) -> str:
        tok = m.group(0)
        if tok in mapping:
            return mapping[tok]
        segs = HUMP.findall(tok)
        if len(segs) > 1 and "".join(segs) == tok:
            return "".join(mapping.get(s, s) for s in segs)
        return tok

    return WORD.sub(fix, java)


class RetrievalBaseline:
    def __init__(self, train_records: Sequence[dict], max_df: float = 0.5):
        self.recs = list(train_records)
        n = len(self.recs)
        if n == 0:
            raise ValueError("retrieval baseline needs training examples")
        self.toks = [words(r["instruction"]) for r in self.recs]
        df: Counter = Counter()
        for t in self.toks:
            df.update({w.lower() for w in t})
        self.idf = {w: math.log((1 + n) / (1 + c)) + 1.0 for w, c in df.items() if c <= max_df * n}
        post: dict[str, list] = {}
        norms = np.zeros(n)
        for i, t in enumerate(self.toks):
            tf = Counter(w.lower() for w in t if w.lower() in self.idf)
            vec = {w: (1 + math.log(c)) * self.idf[w] for w, c in tf.items()}
            norms[i] = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            for w, v in vec.items():
                post.setdefault(w, []).append((i, v))
        self.norms = norms
        self.post = {w: (np.array([i for i, _ in p]), np.array([v for _, v in p])) for w, p in post.items()}

    def nearest(self, instruction: str) -> int:
        tf = Counter(w.lower() for w in words(instruction) if w.lower() in self.post)
        score = np.zeros(len(self.recs))
        for w, c in tf.items():
            idx, val = self.post[w]
            score[idx] += val * ((1 + math.log(c)) * self.idf[w])
        return int(np.argmax(score / self.norms))

    def generate_one(self, instruction: str) -> str:
        r = self.recs[self.nearest(instruction)]
        return substitute(r["java"], words(r["instruction"]), words(instruction))
