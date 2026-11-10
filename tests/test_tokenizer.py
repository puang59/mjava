import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
from helpers import FIX, read_jsonl  # noqa: E402

from mjava.interfaces import SPECIAL_TOKENS, build_generate_source, build_repair_source  # noqa: E402
from mjava.tokenizer import decode, encode, load_tokenizer, train_tokenizer  # noqa: E402


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    out = tmp_path_factory.mktemp("tok") / "tokenizer.json"
    train_tokenizer([FIX / "train.jsonl", FIX / "repair_train.jsonl"], out, vocab_size=500)
    return load_tokenizer(out)


def test_specials_single_ids(tok):
    for i, s in enumerate(SPECIAL_TOKENS):
        assert encode(tok, s) == [i]
    assert (tok.pad_id, tok.bos_id, tok.eos_id, tok.unk_id) == (0, 1, 2, 3)
    assert tok.encode(build_generate_source("make a class"))[0] == tok.gen_id
    ids = tok.encode(build_repair_source("x", "line 1: bad", "class A {}"))
    assert ids[0] == tok.fix_id and tok.err_id in ids and tok.code_id in ids


def test_roundtrip_fixture_java(tok):
    for r in read_jsonl(FIX / "train.jsonl") + read_jsonl(FIX / "repair_train.jsonl"):
        for text in (r["java"], r.get("broken_java", "")):
            assert decode(tok, encode(tok, text)) == text


@pytest.mark.parametrize(
    "text",
    [
        "class A {\n\tint x;\r\n    \n\n        void f() {}   \n}\n",
        "  leading and trailing  ",
        "String s = \"café ☃ \U0001f600\";  // unicode",
        "a\n\n\n\nb",
        "",
        "x  =  y;;;   \t",
    ],
)
def test_roundtrip_edge_whitespace(tok, text):
    assert decode(tok, encode(tok, text)) == text


def test_decode_skips_specials(tok):
    ids = [tok.bos_id] + tok.encode("int x;") + [tok.eos_id, tok.pad_id]
    assert decode(tok, ids) == "int x;"
    assert "<eos>" in decode(tok, ids, skip_special=False)
