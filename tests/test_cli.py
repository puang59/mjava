import hashlib
import shutil
from pathlib import Path

import pytest

from mjava.cli import main

try:
    from test_pipeline_fakes import STUDENT_TABLE, FakeCompiler, JavacCompiler, ScriptedGenerator
except ImportError:
    from tests.test_pipeline_fakes import STUDENT_TABLE, FakeCompiler, JavacCompiler, ScriptedGenerator

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def make(tmp_path, text=None, name="Student.mjava"):
    p = tmp_path / name
    if text is None:
        p.write_bytes((EXAMPLES / "Student.mjava").read_bytes())
    else:
        p.write_bytes(text if isinstance(text, bytes) else text.encode())
    return p


def test_student_example_end_to_end_with_fakes(tmp_path, capsys):
    src = make(tmp_path)
    before = digest(src)
    rc = main([str(src)], generator=ScriptedGenerator(STUDENT_TABLE), compiler=FakeCompiler())
    assert rc == 0
    out = tmp_path / "Student.java"
    assert out.is_file() and "class Student" in out.read_text() and "BUG" not in out.read_text()
    assert digest(src) == before
    assert not (tmp_path / "Student.mjava.review.txt").exists()


def test_output_flag_and_input_untouched(tmp_path):
    src = make(tmp_path)
    before = digest(src)
    out = tmp_path / "sub" / "X.java"
    rc = main([str(src), "-o", str(out)], generator=ScriptedGenerator(STUDENT_TABLE), compiler=FakeCompiler())
    assert rc == 0 and out.is_file() and not (tmp_path / "Student.java").exists()
    assert digest(src) == before


def test_failure_exit_1_writes_review_and_no_java(tmp_path, capsys):
    src = make(tmp_path)
    before = digest(src)
    rc = main([str(src), "--max-retries", "2"],
              generator=ScriptedGenerator(STUDENT_TABLE, fix_after=10**6), compiler=FakeCompiler())
    assert rc == 1
    assert digest(src) == before
    assert not (tmp_path / "Student.java").exists()
    review = (tmp_path / "Student.mjava.review.txt").read_text()
    assert "MANUAL REVIEW" in review and "cannot find symbol BUG" in review and "class Student" in review
    err = capsys.readouterr().err
    assert "manual review" in err and "cannot find symbol BUG" in err


def test_success_removes_stale_review_file(tmp_path):
    src = make(tmp_path)
    (tmp_path / "Student.mjava.review.txt").write_text("old")
    assert main([str(src)], generator=ScriptedGenerator(STUDENT_TABLE), compiler=FakeCompiler()) == 0
    assert not (tmp_path / "Student.mjava.review.txt").exists()


def test_verbose_shows_each_attempt_and_the_fed_back_error(tmp_path, capsys):
    src = make(tmp_path)
    main([str(src), "--verbose"], generator=ScriptedGenerator(STUDENT_TABLE), compiler=FakeCompiler())
    out = capsys.readouterr().out
    assert "attempt 0 [generate]" in out and "attempt 1 [repair]" in out
    assert "compiler error fed back to the model: line 4: cannot find symbol BUG" in out
    assert "result: FAILED" in out and "result: OK" in out


def test_generator_crash_exit_2_original_intact(tmp_path, capsys):
    class Boom:
        def generate(self, sources, config=None):
            raise RuntimeError("model exploded")

    src = make(tmp_path)
    before = digest(src)
    rc = main([str(src)], generator=Boom(), compiler=FakeCompiler())
    assert rc == 2 and digest(src) == before
    assert "model exploded" in capsys.readouterr().err
    assert not (tmp_path / "Student.java").exists()


def test_compiler_crash_mid_run_original_intact(tmp_path):
    class BadCompiler(FakeCompiler):
        def compile(self, source, filename="Main.java"):
            raise OSError("jvm died")

    src = make(tmp_path)
    before = digest(src)
    assert main([str(src)], generator=ScriptedGenerator(STUDENT_TABLE), compiler=BadCompiler()) == 2
    assert digest(src) == before


def test_read_only_input_is_fine(tmp_path):
    src = make(tmp_path)
    src.chmod(0o444)
    try:
        assert main([str(src)], generator=ScriptedGenerator(STUDENT_TABLE), compiler=FakeCompiler()) == 0
    finally:
        src.chmod(0o666)


def test_missing_input_exit_2(tmp_path, capsys):
    assert main([str(tmp_path / "nope.mjava")], generator=object(), compiler=FakeCompiler()) == 2
    assert "not found" in capsys.readouterr().err


def test_missing_checkpoint_clear_error_exit_2(tmp_path, capsys):
    src = make(tmp_path)
    rc = main([str(src), "--checkpoint", str(tmp_path / "no_ckpt")], compiler=FakeCompiler())
    assert rc == 2
    assert "checkpoint directory not found" in capsys.readouterr().err


def test_missing_compiler_or_model_gives_clear_error(tmp_path, capsys, monkeypatch):
    src = make(tmp_path)
    ck = tmp_path / "ckpt"
    ck.mkdir()
    import sys

    monkeypatch.setitem(sys.modules, "mjava.compiler", None)  # import raises ImportError
    assert main([str(src), "--checkpoint", str(ck)]) == 2
    assert "mjava.compiler" in capsys.readouterr().err
    monkeypatch.setitem(sys.modules, "mjava.model.generator", None)
    assert main([str(src), "--checkpoint", str(ck)], compiler=FakeCompiler()) == 2
    assert "mjava.model.generator" in capsys.readouterr().err


def test_negative_retries_exit_2(tmp_path):
    src = make(tmp_path)
    assert main([str(src), "--max-retries", "-1"], generator=object(), compiler=FakeCompiler()) == 2


def test_max_retries_zero_means_single_attempt(tmp_path):
    src = make(tmp_path)
    gen = ScriptedGenerator(STUDENT_TABLE)
    assert main([str(src), "--max-retries", "0"], generator=gen, compiler=FakeCompiler()) == 1
    assert len(gen.calls) == 1


def test_empty_file_and_no_shorthand(tmp_path, capsys):
    src = make(tmp_path, "", "Empty.mjava")
    gen = ScriptedGenerator({})
    assert main([str(src)], generator=gen, compiler=FakeCompiler()) == 0
    assert (tmp_path / "Empty.java").read_text() == "" and not gen.calls
    src2 = make(tmp_path, "class A { }\n", "Plain.mjava")
    assert main([str(src2)], generator=gen, compiler=FakeCompiler()) == 0
    assert (tmp_path / "Plain.java").read_text() == "class A { }\n"
    assert "no shorthand" in capsys.readouterr().err


def test_crlf_input_preserved_and_original_bytes_identical(tmp_path):
    raw = (EXAMPLES / "Student.mjava").read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    src = make(tmp_path, raw)
    assert main([str(src)], generator=ScriptedGenerator(STUDENT_TABLE), compiler=FakeCompiler()) == 0
    assert src.read_bytes() == raw
    assert b"\r\n" in (tmp_path / "Student.java").read_bytes()


def test_keep_temp(tmp_path, capsys):
    src = make(tmp_path)
    assert main([str(src), "--keep-temp"], generator=ScriptedGenerator(STUDENT_TABLE), compiler=FakeCompiler()) == 0
    line = [l for l in capsys.readouterr().err.splitlines() if "temp files kept" in l][0]
    tmpdir = Path(line.split("kept in ")[1])
    assert (tmpdir / "Student.java").is_file()
    shutil.rmtree(tmpdir, ignore_errors=True)


def test_examples_exist_and_parse():
    from mjava.pipeline import parse_mjava

    names = sorted(p.name for p in EXAMPLES.glob("*.mjava"))
    assert "Student.mjava" in names and len(names) >= 3
    for p in EXAMPLES.glob("*.mjava"):
        assert any(b.kind == "shorthand" for b in parse_mjava(p.read_text()))


@pytest.mark.javac
@pytest.mark.skipif(shutil.which("javac") is None, reason="javac not on PATH")
def test_mixed_example_with_real_javac_and_run(tmp_path, capfd):
    src = tmp_path / "Library.mjava"
    shutil.copyfile(EXAMPLES / "Library.mjava", src)
    table = {
        "class Book": ("class Book {\n    String title\n    int pages;\n}",
                       "class Book {\n    String title;\n    int pages;\n}"),
        "enum Color": ("enum Color { RED, GREEN, BLUE }", "enum Color { RED, GREEN, BLUE }"),
    }
    before = digest(src)
    rc = main([str(src), "--run", "--verbose"], generator=ScriptedGenerator(table), compiler=JavacCompiler())
    out = capfd.readouterr().out
    assert rc == 0, out
    assert "hello" in out and "';' expected" in out
    assert digest(src) == before
    java = (tmp_path / "Library.java").read_text()
    assert java.startswith("// Mixes") and java.count("import java.util.List;") == 1
    assert "class Main" in java


@pytest.mark.javac
def test_cli_with_project_compiler_if_available(tmp_path):
    comp_mod = pytest.importorskip("mjava.compiler")
    if shutil.which("javac") is None:
        pytest.skip("no javac")
    src = make(tmp_path)
    table = {"class Student": ("class Student {\n    String name\n}", "class Student {\n    String name;\n}")}
    comp = comp_mod.get_compiler()
    try:
        assert main([str(src)], generator=ScriptedGenerator(table), compiler=comp) == 0
    finally:
        comp.close()
