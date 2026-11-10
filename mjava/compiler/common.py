"""Helpers shared by the compiler implementations."""
from __future__ import annotations

import re

from mjava.interfaces import CompileResult, Diagnostic

_TYPE_RE = re.compile(
    r"\b(public\s+)?(?:(?:abstract|final|static|sealed)\s+)*(?:class|interface|enum|record|@interface)\s+([A-Za-z_$][\w$]*)"
)


def derive_filename(source: str, default: str = "Main.java") -> str:
    """File name matching the first top-level type (a public one is preferred)."""
    first = None
    for m in _TYPE_RE.finditer(source):
        if m.group(1):
            return m.group(2) + ".java"
        if first is None:
            first = m.group(2)
    return (first + ".java") if first else default


def result_from_json(obj: dict) -> CompileResult:
    diags = tuple(
        Diagnostic(
            kind=str(d.get("kind", "ERROR")),
            line=int(d.get("line", -1)),
            column=int(d.get("column", -1)),
            code=str(d.get("code", "")),
            message=str(d.get("message", "")),
        )
        for d in obj.get("diagnostics", [])
    )
    return CompileResult(ok=bool(obj.get("ok", False)), diagnostics=diags)
