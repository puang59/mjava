"""Test doubles shared by the pipeline and CLI tests (no tests in here)."""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from mjava.interfaces import CompileResult, Diagnostic


class ScriptedGenerator:
    """Broken answer first; a fixed answer when the source starts with <fix>.

    ``table`` maps an instruction substring -> (broken, fixed).  Calls are
    recorded in ``calls`` as (source, config).  Sampling returns ``fixed``
    plus numbered variants so candidates are distinct.
    """

    def __init__(self, table, fix_after: int = 1):
        self.table = table
        self.calls: list[tuple[str, object]] = []
        self.fix_after = fix_after  # <fix> calls per instruction needed before the fixed answer
        self._fix_seen: dict[str, int] = {}

    def _pair(self, source):
        for key, pair in self.table.items():
            if key in source:
                return key, pair
        raise KeyError(f"no script for source: {source!r}")

    def generate(self, sources, config=None):
        out = []
        for s in sources:
            self.calls.append((s, config))
            key, (broken, fixed) = self._pair(s)
            n = getattr(config, "num_return_sequences", 1) if config else 1
            if s.startswith("<fix>"):
                self._fix_seen[key] = self._fix_seen.get(key, 0) + 1
                good = self._fix_seen[key] >= self.fix_after
                base = fixed if good else broken
                if n > 1:
                    out.append([base] + [base + f"\n// v{i}" for i in range(1, n)])
                else:
                    out.append([base])
            else:
                out.append([broken])
        return out

    def sources(self):
        return [s for s, _ in self.calls]


class FakeCompiler:
    """Tiny line-accurate fake.  Errors: a line containing BUG (cannot find
    symbol), unbalanced braces (line of the last line), and `uses Foo` lines
    where `class Foo` / `enum Foo` is not defined anywhere in the source."""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []
        self.closed = False

    def compile(self, source, filename="Main.java"):
        self.calls.append((source, filename))
        diags = []
        lines = source.split("\n")
        for i, l in enumerate(lines, 1):
            if "BUG" in l:
                diags.append(Diagnostic("ERROR", i, 1, "compiler.err.cant.resolve.location", "cannot find symbol BUG"))
            m = re.search(r"uses (\w+)", l)
            if m and not re.search(rf"\b(class|enum|interface|record)\s+{m.group(1)}\b", source):
                diags.append(
                    Diagnostic("ERROR", i, 1, "compiler.err.cant.resolve", f"cannot find symbol {m.group(1)}")
                )
        if source.count("{") != source.count("}"):
            diags.append(Diagnostic("ERROR", len(lines), 1, "compiler.err.premature.eof", "reached end of file"))
        return CompileResult(not diags, tuple(diags))

    def compile_many(self, sources):
        return [self.compile(s) for s in sources]

    def close(self):
        self.closed = True


class JavacCompiler:
    """Minimal real compiler via the `javac` executable."""

    _ERR = re.compile(r"^(?P<file>.+?\.java):(?P<line>\d+): (?P<kind>error|warning): (?P<msg>.*)$")

    def compile(self, source, filename="Main.java"):
        d = Path(tempfile.mkdtemp(prefix="mjava-test-"))
        try:
            (d / filename).write_text(source, encoding="utf-8", newline="")
            r = subprocess.run(
                ["javac", "-proc:none", "-Xlint:none", "-d", str(d / "out"), str(d / filename)],
                capture_output=True, text=True,
            )
        finally:
            pass
        diags = []
        lines = r.stderr.splitlines()
        for i, l in enumerate(lines):
            m = self._ERR.match(l)
            if m:
                col = -1
                if i + 2 < len(lines) and lines[i + 2].strip() == "^":
                    col = lines[i + 2].index("^") + 1
                diags.append(Diagnostic(m["kind"].upper(), int(m["line"]), col, "javac", m["msg"]))
        shutil.rmtree(d, ignore_errors=True)
        ok = r.returncode == 0
        if not ok and not any(x.kind == "ERROR" for x in diags):
            diags.append(Diagnostic("ERROR", -1, -1, "javac", r.stderr.strip()[:200]))
        return CompileResult(ok, tuple(diags))

    def compile_many(self, sources):
        return [self.compile(s) for s in sources]

    def close(self):
        pass


STUDENT_FIXED = (
    "class Student {\n    String name;\n    int age;\n    double gpa;\n}"
)
STUDENT_BROKEN = (
    "import java.util.List;\nclass Student {\n    String name;\n    BUG age;\n    double gpa;\n}"
)
STUDENT_TABLE = {"class Student": (STUDENT_BROKEN, "import java.util.List;\n" + STUDENT_FIXED)}
