import json
import shutil

import pytest

from mjava.interfaces import GenerationConfig
from mjava.pipeline import SelfCorrector

try:
    from test_pipeline_fakes import (
        STUDENT_TABLE, FakeCompiler, JavacCompiler, ScriptedGenerator,
    )
except ImportError:  # package-style test discovery
    from tests.test_pipeline_fakes import (
        STUDENT_TABLE, FakeCompiler, JavacCompiler, ScriptedGenerator,
    )

SLIDE = (
    "// Custom syntax a user would actually write --\n"
    "// natural language instructions, not new syntax rules\n"
    "Create a class Student with fields: name as String,\n"
    "  age as int, gpa as double.\n"
)


def run(text, table=STUDENT_TABLE, retries=3, fix_after=1, compiler=None):
    gen = ScriptedGenerator(table, fix_after=fix_after)
    comp = compiler or FakeCompiler()
    res = SelfCorrector(gen, comp, max_retries=retries).run(text)
    return res, gen, comp


def test_student_fixed_on_first_retry_and_error_is_fed_back():
    res, gen, _ = run(SLIDE)
    assert res.ok
    (b,) = res.blocks
    assert b.success and b.succeeded_at == 1 and b.retries_used == 1
    a0, a1 = b.attempts
    assert (a0.strategy, a1.strategy) == ("generate", "repair")
    assert a0.source.startswith("<gen> Create a class Student")
    # fed-back error uses the line number inside the block's own code (BUG is on line 4 there)
    assert a1.source.startswith("<fix> Create a class Student")
    assert "<err> line 4: cannot find symbol BUG <code> import java.util.List;" in a1.source
    assert "BUG" not in res.java and "class Student" in res.java
    assert res.java.startswith("// Custom syntax")
    assert res.java.count("import java.util.List;") == 1


def test_trace_is_json_serialisable_and_shaped():
    res, _, _ = run(SLIDE)
    d = json.loads(json.dumps(res.to_dict()))
    assert d["ok"] is True and d["max_retries"] == 3
    blk = d["blocks"][0]
    assert {"success", "succeeded_at", "retries_used", "attempts", "instruction"} <= set(blk)
    att = blk["attempts"][0]
    assert {"index", "strategy", "phase", "source", "config", "code", "ok", "diagnostics", "error_text"} <= set(att)
    assert att["diagnostics"][0]["message"] == "cannot find symbol BUG"
    assert [o["kind"] for o in d["overview"]] == ["comment", "shorthand"]


def test_success_within_k():
    res, *_ = run(SLIDE)
    b = res.blocks[0]
    assert not b.success_within(0) and b.success_within(1) and b.success_within(3)


def test_first_attempt_success_makes_no_repair_call():
    table = {"class Student": ("class Student { }", "class Student { }")}
    res, gen, _ = run(SLIDE, table)
    assert res.ok and res.blocks[0].retries_used == 0
    assert len(gen.calls) == 1


@pytest.mark.parametrize("retries", [0, 1, 2, 3, 5])
def test_never_fixed_terminates_within_limit(retries):
    res, gen, comp = run(SLIDE, retries=retries, fix_after=10**6)
    assert not res.ok
    b = res.blocks[0]
    assert len(b.attempts) <= retries + 1
    assert b.retries_used <= retries
    assert not b.success and b.succeeded_at is None
    assert res.error_text  # remaining errors reported
    assert res.java  # last attempt still available for the review file
    # bounded generator work: one batched call + at most 2 calls per retry (repair + escalation)
    assert len(gen.calls) <= 1 + 2 * retries


def test_never_retries_with_a_candidate_already_tried():
    # a generator that always answers the same broken code
    gen = ScriptedGenerator({"class Student": ("class Student {\n BUG x;\n}", "class Student {\n BUG x;\n}")}, fix_after=1)
    res = SelfCorrector(gen, FakeCompiler(), max_retries=5).run(SLIDE)
    codes = [a.code for a in res.blocks[0].attempts if a.code is not None]
    norm = [" ".join(c.split()) for c in codes]
    assert len(norm) == len(set(norm))
    assert not res.ok
    # escalation to sampling produced distinct variants, then the block was exhausted or budget used
    assert any(a.escalated_from == "repair" for a in res.blocks[0].attempts[1:])


def test_exhausted_block_stops_when_generator_has_nothing_new():
    class Stuck:
        def generate(self, sources, config=None):
            return [["class Student {\n BUG x;\n}"] for _ in sources]

    res = SelfCorrector(Stuck(), FakeCompiler(), max_retries=5).run(SLIDE)
    b = res.blocks[0]
    assert not res.ok
    assert b.attempts[-1].code is None and "exhausted" in b.attempts[-1].note
    assert len(b.attempts) == 2  # attempt 0, then one attempt discovering nothing new
    assert b.last_coded().index == 0


def test_strategy_escalation_to_sampling():
    res, gen, _ = run(SLIDE, fix_after=4)
    b = res.blocks[0]
    assert res.ok
    strategies = [a.strategy for a in b.attempts]
    assert strategies[0] == "generate" and strategies[-1] == "sample"
    sample_calls = [c for s, c in gen.calls if c is not None and c.do_sample]
    assert sample_calls and sample_calls[0].num_return_sequences == 4


def test_sampling_takes_first_candidate_that_compiles():
    class G:
        def generate(self, sources, config=None):
            if config is not None and config.do_sample:
                return [["class A {\n BUG a;\n}", "class A {\n int ok;\n}", "class A {\n int later;\n}"]]
            return [["class A {\n BUG a;\n}"] if s.startswith("<gen>") else ["class A {\n BUG a;\n}"] for s in sources]

    res = SelfCorrector(G(), FakeCompiler(), max_retries=3).run("Make class A.")
    b = res.blocks[0]
    assert res.ok and b.attempts[-1].strategy == "sample"
    assert "int ok;" in res.java and "later" not in res.java


def test_empty_model_output_is_a_failure_then_repaired():
    class G:
        def __init__(self):
            self.n = 0

        def generate(self, sources, config=None):
            self.n += 1
            return [[""] if self.n == 1 else ["class A { }"] for _ in sources]

    res = SelfCorrector(G(), FakeCompiler(), max_retries=2).run("Make class A.")
    assert res.ok
    a0 = res.blocks[0].attempts[0]
    assert not a0.ok and a0.diagnostics[0]["code"] == "mjava.empty"


def test_markdown_fences_are_stripped():
    class G:
        def generate(self, sources, config=None):
            return [["```java\nclass A { }\n```"] for _ in sources]

    res = SelfCorrector(G(), FakeCompiler()).run("Make class A.")
    assert res.ok and "```" not in res.java


def test_only_the_failing_block_is_retried():
    text = "Make class A.\n\nMake class B.\n\nMake class C."
    table = {
        "class A": ("class A { }", "class A { }"),
        "class B": ("class B {\n BUG b;\n}", "class B {\n int b;\n}"),
        "class C": ("class C { }", "class C { }"),
    }
    res, gen, _ = run(text, table)
    assert res.ok
    repairs = [s for s in gen.sources() if s.startswith("<fix>")]
    assert len(repairs) == 1 and "class B" in repairs[0]
    assert [b.retries_used for b in res.blocks] == [0, 1, 0]
    # the three initial generates came in one batched call
    assert len(gen.calls[0:3]) == 3 and gen.sources()[0].startswith("<gen>")


def test_forward_reference_between_shorthand_blocks_is_deferred_not_retried():
    text = "Make class A that uses B.\n\nMake class B."
    table = {
        "class A": ("class A {\n // uses B\n}", "x"),
        "class B": ("class B { }", "class B { }"),
    }
    res, gen, _ = run(text, table)
    assert res.ok
    assert not [s for s in gen.sources() if s.startswith("<fix>")]
    assert all(b.success and b.retries_used == 0 for b in res.blocks)


def test_file_level_error_maps_to_the_owning_block():
    # B mentions an undefined symbol: deferred in phase 1 (it might live in a sibling), blamed in the
    # file phase by output line number, and only B is retried; the error is fed back B-relative.
    text = "Make class A.\n\nMake class B."
    table = {
        "class A": ("class A { }", "class A { }"),
        "class B": ("class B {\n // uses Z\n}", "class B {\n // uses A\n}"),
    }
    res, gen, _ = run(text, table)
    assert res.ok
    a, b = res.blocks
    assert a.retries_used == 0 and all(x.phase == "block" for x in a.attempts)
    assert [x.phase for x in b.attempts] == ["block", "file"]
    assert b.attempts[0].diagnostics and not b.attempts[0].ok
    assert "<err> line 2: cannot find symbol Z" in b.attempts[1].source
    assert len([s for s in gen.sources() if s.startswith("<fix>")]) == 1


def test_error_in_handwritten_java_is_not_retryable():
    text = "class Keep {\n BUG k;\n}\n\nMake class A."
    table = {"class A": ("class A { }", "class A { }")}
    res, gen, _ = run(text, table)
    assert not res.ok
    assert not [s for s in gen.sources() if s.startswith("<fix>")]
    assert res.blocks[0].success  # the shorthand block itself is fine
    assert any("BUG" in d.message for d in res.file_diagnostics)


def test_imports_hoisted_and_deduplicated_across_blocks():
    text = "import java.util.List;\n\nMake class A.\n\nMake class B."
    table = {
        "class A": ("import java.util.List;\nimport java.util.Map;\nclass A { }", ""),
        "class B": ("import java.util.Map;\nclass B { }", ""),
    }
    res, _, _ = run(text, table)
    assert res.ok
    assert res.java.count("import java.util.List;") == 1
    assert res.java.count("import java.util.Map;") == 1
    first_class = res.java.index("class A")
    assert res.java.rindex("import") < first_class


def test_empty_file_and_comment_only_files_do_not_compile_or_generate():
    for text in ["", "\n\n", "// nothing here\n"]:
        res, gen, comp = run(text)
        assert res.ok and res.blocks == [] and not gen.calls and not comp.calls


def test_file_without_shorthand_is_compiled_as_is():
    res, gen, comp = run("class A { }\n")
    assert res.ok and not gen.calls and len(comp.calls) == 1
    assert res.java == "class A { }\n"
    res, *_ = run("class A {\n BUG x;\n}\n")
    assert not res.ok and res.blocks == []


def test_crlf_input_gives_crlf_output():
    res, *_ = run(SLIDE.replace("\n", "\r\n"))
    assert res.ok
    assert "\r\n" in res.java and "\n" not in res.java.replace("\r\n", "")


def test_gen_marker_block_is_expanded():
    table = {"counter": ("class Counter { }", "class Counter { }")}
    res, gen, _ = run("@gen a counter\nclass", table)
    assert res.ok and gen.sources()[0] == "<gen> a counter class"


def test_run_file_reads_binary_and_correct_instruction(tmp_path):
    p = tmp_path / "S.mjava"
    p.write_bytes(SLIDE.encode())
    gen = ScriptedGenerator(STUDENT_TABLE)
    sc = SelfCorrector(gen, FakeCompiler())
    assert sc.run_file(p).ok
    tr = sc.correct_instruction("Create a class Student with name.")
    assert tr.success and tr.succeeded_at == 1


def test_generator_exception_propagates():
    class Boom:
        def generate(self, sources, config=None):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        SelfCorrector(Boom(), FakeCompiler()).run("Make class A.")


def test_on_attempt_callback_and_config_passed():
    seen = []
    gen = ScriptedGenerator(STUDENT_TABLE)
    SelfCorrector(gen, FakeCompiler(), on_attempt=lambda t, a: seen.append(a.index)).run(SLIDE)
    assert seen == [0, 1]
    assert isinstance(gen.calls[0][1], GenerationConfig)


# ---------------------------------------------------------------- real javac
@pytest.mark.javac
@pytest.mark.skipif(shutil.which("javac") is None, reason="javac not on PATH")
def test_with_real_javac_missing_semicolon_is_repaired():
    table = {
        "class Student": (
            "class Student {\n    String name\n    int age;\n}",
            "class Student {\n    String name;\n    int age;\n}",
        )
    }
    res, gen, _ = run(SLIDE, table, compiler=JavacCompiler())
    assert res.ok
    a0 = res.blocks[0].attempts[0]
    assert not a0.ok and "';' expected" in a0.error_text
    assert "';' expected" in res.blocks[0].attempts[1].source


@pytest.mark.javac
@pytest.mark.skipif(shutil.which("javac") is None, reason="javac not on PATH")
def test_with_real_javac_mixed_file_maps_error_to_block():
    text = (
        "Make Book.\n\nMake Shelf.\n\n"
        "class Main {\n    public static void main(String[] a) { System.out.println(new Shelf().b.title); }\n}\n"
    )
    table = {
        "Book": ("class Book { String title; }", "class Book { String title; }"),
        "Shelf": ("class Shelf { Bok b; }", "class Shelf { Book b; }"),
    }
    gen = ScriptedGenerator(table)
    res = SelfCorrector(gen, JavacCompiler(), max_retries=3).run(text)
    assert res.ok, res.error_text
    repairs = [s for s in gen.sources() if s.startswith("<fix>")]
    assert len(repairs) == 1 and "Shelf" in repairs[0]
    assert [b.retries_used for b in res.blocks] == [0, 1]


@pytest.mark.javac
def test_with_project_compiler_if_available():
    comp_mod = pytest.importorskip("mjava.compiler")
    if shutil.which("javac") is None:
        pytest.skip("no javac")
    comp = comp_mod.get_compiler()
    try:
        table = {
            "class Student": (
                "class Student {\n    String name\n    int age;\n}",
                "class Student {\n    String name;\n    int age;\n}",
            )
        }
        res, *_ = run(SLIDE, table, compiler=comp)
        assert res.ok and res.blocks[0].retries_used == 1
    finally:
        comp.close()
