"""SubprocessCompiler: one ``javac`` process per compile (slow fallback)."""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

from mjava.compiler.common import derive_filename
from mjava.interfaces import CompileResult, Diagnostic

_DIAG = re.compile(r"^(?P<file>.+?\.java):(?P<line>\d+):\s*(?P<kind>error|warning):\s*(?P<msg>.*)$")
_COUNT = re.compile(r"^\d+ (errors?|warnings?)$")


def parse_javac_output(text: str) -> tuple[Diagnostic, ...]:
    diags: list[Diagnostic] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = _DIAG.match(lines[i])
        if not m:
            i += 1
            continue
        msg = m.group("msg").strip()
        col = -1
        j = i + 1
        extra: list[str] = []
        while j < len(lines) and not _DIAG.match(lines[j]) and not _COUNT.match(lines[j]):
            s = lines[j]
            if s.strip() == "^":
                col = s.index("^") + 1
            elif s.strip().startswith("symbol:"):
                extra.append(s.strip())
            j += 1
        if extra:
            msg += " " + " ".join(extra)
        msg = re.sub(r"\s+", " ", msg.replace("java.lang.", ""))
        diags.append(Diagnostic("ERROR" if m.group("kind") == "error" else "WARNING",
                                int(m.group("line")), col, "", msg))
        i = j
    return tuple(diags)


class SubprocessCompiler:
    def __init__(self, timeout: float = 30.0):
        self.javac = shutil.which("javac")
        if not self.javac:
            raise RuntimeError("javac not found: install a JDK (11 or newer) and put javac on PATH")
        self.timeout = timeout

    def compile(self, source: str, filename: str | None = None) -> CompileResult:
        if filename is None or filename == "Main.java":
            filename = derive_filename(source, filename or "Main.java")
        with tempfile.TemporaryDirectory(prefix="mjava_") as d:
            f = Path(d) / filename
            f.write_text(source, encoding="utf-8")
            try:
                p = subprocess.run(
                    [self.javac, "-encoding", "UTF-8", "-proc:none", "-Xlint:none", "-J-Duser.language=en",
                     "-d", d, str(f)],
                    capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=self.timeout)
            except subprocess.TimeoutExpired:
                return CompileResult(False, (Diagnostic("ERROR", -1, -1, "javac.timeout", "javac timed out"),))
            diags = parse_javac_output(p.stderr)
            ok = p.returncode == 0
            if not ok and not any(x.kind == "ERROR" for x in diags):
                diags += (Diagnostic("ERROR", -1, -1, "", p.stderr.strip()[:200] or "javac failed"),)
            return CompileResult(ok, diags)

    def compile_many(self, sources: Sequence[str]) -> list[CompileResult]:
        return [self.compile(s) for s in sources]

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()
