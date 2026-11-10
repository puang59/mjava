import json
import random
import re

import pytest

from mjava.data import names
from mjava.data.build import build, make_example
from mjava.data.conformance import missing_elements, spec_conformance
from mjava.data.mutate import MUTATIONS, make_broken
from mjava.data.render_java import render_java
from mjava.data.render_nl import SLIDE_STYLE, nl_data_class, render_nl
from mjava.data.spec import FAMILIES, new_spec, new_type, make_ctor, param, sample_spec, accessor_methods
from mjava.interfaces import RepairExample, Example


@pytest.fixture(scope="module")
def comp():
    from mjava.compiler import get_compiler

    c = get_compiler("auto")
    yield c
    c.close()


def test_pools_clean():
    names.check_pools()


def test_slide_sentence_exact_and_java():
    fields = [param("name", "String"), param("age", "int"), param("gpa", "double")]
    t = new_type("class", "Student", fields=fields, features={"ctor": True})
    t["ctor"] = make_ctor(fields)
    spec = new_spec("data_class", None)
    spec["types"] = [t]
    sentence = nl_data_class(spec, random.Random(0), style=SLIDE_STYLE)
    assert sentence == "Create a class Student with fields: name as String, age as int, gpa as double."
    java = render_java(spec)
    assert "private String name;" in java and "private int age;" in java and "private double gpa;" in java
    assert "this.name = name;" in java and "public Student(String name, int age, double gpa)" in java
    assert spec_conformance(spec, java) == 1.0


def test_slide_sentence_is_in_distribution():
    """The exact slide sentence must be produced by the ordinary sampler, not only by the special style."""
    fields = [param("name", "String"), param("age", "int"), param("gpa", "double")]
    t = new_type("class", "Student", fields=fields, features={"ctor": True})
    t["ctor"] = make_ctor(fields)
    spec = new_spec("data_class", None)
    spec["types"] = [t]
    target = "Create a class Student with fields: name as String, age as int, gpa as double."
    seen = set()
    for i in range(4000):
        seen.add(render_nl(spec, random.Random(i)))
    assert target in seen
    assert len(seen) > 400  # genuinely varied phrasing


@pytest.mark.javac
@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_family_reference_compiles_and_conforms(family, comp):
    rng = random.Random(7)
    for i in range(60):
        spec = sample_spec(rng, heldout=(i % 2 == 0), family=family)
        java = render_java(spec)
        r = comp.compile(java)
        assert r.ok, (java, r.error_text())
        assert spec_conformance(spec, java) == 1.0, missing_elements(spec, java)
        assert spec["family"] == family
        assert spec["heldout"] == (i % 2 == 0)
        assert render_nl(spec, rng).strip()


def test_removing_member_lowers_conformance():
    rng = random.Random(3)
    checked = 0
    for fam in ("data_class", "class_methods", "static_utils", "enum", "interface_impl", "inheritance", "abstract_class"):
        for _ in range(10):
            spec = sample_spec(rng, False, fam)
            java = render_java(spec)
            lines = java.split("\n")
            # drop each line containing a method/constructor/field declaration and expect < 1.0 at least once
            lowered = False
            for i, ln in enumerate(lines):
                if re.search(r"\b(private|protected|public)\b", ln) or ln.strip().startswith("enum"):
                    mutated = "\n".join(lines[:i] + lines[i + 1:])
                    if spec_conformance(spec, mutated) < 1.0:
                        lowered = True
            assert lowered, java
            assert spec_conformance(spec, "class A {}") < 1.0
            checked += 1
    assert checked


def test_conformance_ignores_comments_and_whitespace():
    rng = random.Random(4)
    spec = sample_spec(rng, False, "class_methods")
    java = render_java(spec)
    noisy = "/* header */\n" + re.sub(r" ", "  ", java).replace("{", "{ // c\n")
    assert spec_conformance(spec, noisy) == 1.0
    assert spec_conformance(spec, "// " + java.replace("\n", "\n// ")) < 1.0


def _all_held() -> set:
    from mjava.data.spec import held_words

    return held_words()


def test_heldout_names_only_in_flagged_examples():
    held = _all_held()
    assert len(held) > 150
    rng = random.Random(11)
    for i in range(500):
        h = i % 2 == 0
        spec = sample_spec(rng, h)
        ids = set(re.findall(r"[A-Za-z_]\w*", render_java(spec)))
        # method names are verb+noun compounds, so also split camelCase pieces
        pieces = ids | {p for x in ids for p in re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])", x)}
        pieces |= {p[:1].upper() + p[1:] for p in pieces}
        if h:
            assert (pieces & held) or spec["held_ids"], java_of(spec)
        else:
            assert not (pieces & held), pieces & held
            assert not spec["held_ids"]


def java_of(spec):
    return render_java(spec)


@pytest.mark.javac
def test_mutants_really_fail(comp):
    rng = random.Random(5)
    n = 0
    kinds = set()
    for i in range(60):
        spec = sample_spec(rng, False)
        java = render_java(spec)
        r = make_broken(java, rng, comp)
        if r is None:
            continue
        broken, err, ks = r
        assert broken != java and not comp.compile(broken).ok
        assert err and comp.compile(broken).error_text() == err
        assert comp.compile(java).ok
        kinds.update(ks)
        n += 1
    assert n >= 50
    assert len(kinds) >= 8


@pytest.mark.javac
def test_each_mutation_kind_can_fail(comp):
    rng = random.Random(6)
    ok_kinds = set()
    for _ in range(300):
        spec = sample_spec(rng, False)
        java = render_java(spec)
        for k, fn in MUTATIONS.items():
            out = fn(java, rng)
            if out and not comp.compile(out).ok:
                ok_kinds.add(k)
    assert ok_kinds == set(MUTATIONS), set(MUTATIONS) - ok_kinds


@pytest.mark.javac
def test_make_example_deterministic(comp):
    a = make_example(0, "test", 5, compiler=comp)
    b = make_example(0, "test", 5, compiler=comp)
    assert a == b
    assert make_example(1, "test", 5, compiler=comp) != a


@pytest.mark.javac
@pytest.mark.slow
def test_build_small_and_deterministic(tmp_path):
    args = dict(n_train=60, n_val=10, n_test=20, n_repair_train=20, n_repair_val=5, n_repair_test=8, seed=3, log=lambda *_: None)
    m1 = build(tmp_path / "a", workers=1, **args)
    build(tmp_path / "b", workers=2, **args)  # worker count must not change the output
    for f in sorted(p.name for p in (tmp_path / "a").iterdir()):
        if f == "meta.json":
            continue
        assert (tmp_path / "a" / f).read_bytes() == (tmp_path / "b" / f).read_bytes(), f
    assert m1["sizes"] == {"test": 20, "val": 10, "train": 60}

    def load(name, cls):
        return [cls(**json.loads(l)) for l in (tmp_path / "a" / name).read_text(encoding="utf-8").splitlines()]

    tr, va, te = (load(f"{s}.jsonl", Example) for s in ("train", "val", "test"))
    instr = [e.instruction for e in tr + va + te]
    assert len(instr) == len(set(instr))  # no duplicate instructions, also across splits
    assert not {e.java for e in tr} & ({e.java for e in te} | {e.java for e in va})
    assert any(e.spec["heldout"] for e in te) and not any(e.spec["heldout"] for e in tr + va)
    assert all(e.spec["family"] for e in tr + va + te)
    rtr = load("repair_train.jsonl", RepairExample)
    assert len(rtr) == 20 and all(r.broken_java != r.java and r.error_text for r in rtr)
    corpus = (tmp_path / "a" / "corpus_train.txt").read_text(encoding="utf-8")
    assert "<|endoftext|>" in corpus


def test_synth_identifiers_partition_and_validity():
    from mjava.data import synth
    from mjava.data.names import FORBIDDEN_TYPE_NAMES, JAVA_RESERVED

    rng = random.Random(5)
    seen = set()
    for i in range(1500):
        held = i % 2 == 0
        up = i % 3 == 0
        w = synth.make(rng, up, held)
        assert w.isidentifier() and w[0].isupper() == up
        assert synth.is_held(w) == held
        assert w not in JAVA_RESERVED and w not in FORBIDDEN_TYPE_NAMES
        seen.add(w)
    assert len(seen) > 1000  # an effectively unlimited supply, not a small pool
    c = synth.make_upper_const(rng, False)
    assert c == c.upper() and not synth.is_held(c)
    # deterministic for a given rng state
    assert synth.make(random.Random(9), True, False) == synth.make(random.Random(9), True, False)


def test_synth_names_are_copyable_from_the_instruction():
    """Every class / field / method name a generated example defines must be recoverable from its instruction."""
    rng = random.Random(21)
    skip = {"toString", "equals", "hashCode", "main", "o", "args"}
    for i in range(400):
        spec = sample_spec(rng, heldout=(i % 2 == 0))
        ins = render_nl(spec, rng).lower()

        def walk(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    if k == "name" and isinstance(v, str):
                        yield v
                    else:
                        yield from walk(v)
            elif isinstance(o, list):
                for x in o:
                    yield from walk(x)

        for n in set(walk(spec["types"])) - skip:
            if n.startswith(("get", "set", "is")) and n[3:4].isupper() or n.startswith("is") and n[2:3].isupper():
                continue  # accessors follow from the field name
            assert n.lower() in ins, (n, ins)


def test_standalone_interface_family_member_exists():
    rng = random.Random(3)
    n = 0
    for _ in range(400):
        spec = sample_spec(rng, False, family="interface_impl")
        if len(spec["types"]) == 1:
            n += 1
            assert spec["types"][0]["kind"] == "interface" and spec["types"][0]["methods"]
    assert n > 30
