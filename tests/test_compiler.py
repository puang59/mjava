import os
import threading
import time

import pytest

from mjava.compiler import SubprocessCompiler, get_compiler
from mjava.compiler.common import derive_filename
from mjava.compiler.subprocess_compiler import parse_javac_output
from mjava.interfaces import JavaCompiler

pytestmark = pytest.mark.javac

GOOD = "class A { int f(int a) { return a + 1; } }"
BAD_SEMI = "class A { int f() { return 1 } }"
BAD_TYPE = 'class A { void f() { int x = "a"; y = 2; } }'


@pytest.fixture(scope="module")
def comp():
    c = get_compiler("server")
    yield c
    c.close()


def test_protocol(comp):
    assert isinstance(comp, JavaCompiler)


def test_ok(comp):
    r = comp.compile(GOOD)
    assert r.ok and not r.diagnostics


def test_syntax_error(comp):
    r = comp.compile(BAD_SEMI)
    assert not r.ok
    d = r.errors[0]
    assert d.kind == "ERROR" and d.line == 1 and d.column > 0 and "expected" in d.message
    assert "line 1" in r.error_text()


def test_multiple_errors_single_line_messages(comp):
    r = comp.compile(BAD_TYPE)
    assert not r.ok and len(r.errors) == 2
    assert all("\n" not in d.message for d in r.errors)
    assert any("cannot find symbol" in d.message and "y" in d.message for d in r.errors)
    assert r.errors[0].code.startswith("compiler.err")


def test_public_class_filename_derived(comp):
    assert comp.compile("public class Foo { }").ok
    assert comp.compile("import java.util.List;\npublic class Bar { List<String> x; }").ok
    assert derive_filename("class A {} public class B {}") == "B.java"
    assert derive_filename("int x;") == "Main.java"


def test_unicode_and_escapes(comp):
    r = comp.compile('class U { String s = "héllo \\n \\"q\\" 中"; }')
    assert r.ok
    r = comp.compile('class U { String s = "é"; int x = "中"; }')
    assert not r.ok


def test_compile_many_order(comp):
    rs = comp.compile_many([GOOD, BAD_SEMI, GOOD])
    assert [r.ok for r in rs] == [True, False, True]


def test_threadsafe(comp):
    results = []

    def work(i):
        src = GOOD if i % 2 == 0 else BAD_SEMI
        results.append((i % 2 == 0, comp.compile(src).ok))

    ts = [threading.Thread(target=work, args=(i,)) for i in range(16)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(results) == 16 and all(a == b for a, b in results)


def test_restart_after_jvm_dies():
    c = get_compiler("server")
    try:
        assert c.compile(GOOD).ok
        c._proc.kill()
        c._proc.wait()
        assert c.compile(GOOD).ok
        assert c.restarts >= 1
    finally:
        c.close()


def test_timeout_restarts_and_reports():
    c = get_compiler("server")
    try:
        c.timeout = 0.0001  # impossible deadline: must restart and return a failure, not hang
        r = c.compile(GOOD)
        assert not r.ok and r.errors[0].code == "server.timeout"
        c.timeout = 20
        assert c.compile(GOOD).ok
    finally:
        c.close()


def test_close_terminates_jvm():
    c = get_compiler("server")
    p = c._proc
    assert p.poll() is None
    c.close()
    assert p.poll() is not None
    with pytest.raises(RuntimeError):
        c.compile(GOOD)


def test_context_manager():
    with get_compiler("server") as c:
        p = c._proc
        assert c.compile(GOOD).ok
    assert p.poll() is not None


def test_subprocess_agrees_with_server(comp):
    sp = SubprocessCompiler()
    for src in (GOOD, BAD_SEMI, BAD_TYPE):
        a, b = comp.compile(src), sp.compile(src)
        assert a.ok == b.ok
        assert [d.line for d in a.errors] == [d.line for d in b.errors]
        assert [d.message for d in a.errors] == [d.message for d in b.errors]


def test_parse_javac_output():
    out = "A.java:3: error: ';' expected\n  int x = 1\n           ^\n1 error\n"
    (d,) = parse_javac_output(out)
    assert d.line == 3 and d.column == 12 and d.message == "';' expected"


def test_get_compiler_auto(comp):
    with get_compiler("auto") as c:
        assert c.compile(GOOD).ok


@pytest.mark.skipif(not os.environ.get("MJAVA_PERF"), reason="timing assertion; set MJAVA_PERF=1 to measure (flaky on a busy machine)")
def test_throughput_floor(comp):
    comp.compile(GOOD)
    n = 100
    t = time.time()
    for i in range(n):
        assert comp.compile(f"class T{i} {{ int f(int a) {{ return a + {i}; }} }}").ok
    rate = n / (time.time() - t)
    assert rate > 40, rate  # target is >=100/s on the dev machine; keep the floor loose for CI noise
