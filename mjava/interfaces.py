"""Shared contracts between workstreams.

Every package in ``mjava`` talks to the others only through the types defined
here. Do not change a signature in this file without updating PLAN.md and every
implementer/caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

# --------------------------------------------------------------------------
# Model input format
# --------------------------------------------------------------------------
# The encoder sees one of two source layouts, built with the special tokens
# below. They are part of the tokenizer vocabulary as single tokens.
#
#   generate:  <gen> {instruction}
#   repair:    <fix> {instruction} <err> {error_text} <code> {broken_java}
#
# The decoder always produces plain Java source: <bos> {java} <eos>

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
GEN, FIX, ERR, CODE = "<gen>", "<fix>", "<err>", "<code>"
SPECIAL_TOKENS = [PAD, BOS, EOS, UNK, GEN, FIX, ERR, CODE]

MAX_ERROR_CHARS = 300


def build_generate_source(instruction: str) -> str:
    return f"{GEN} {instruction.strip()}"


def build_repair_source(instruction: str, error_text: str, broken_java: str) -> str:
    error_text = error_text.strip()[:MAX_ERROR_CHARS]
    return f"{FIX} {instruction.strip()} {ERR} {error_text} {CODE} {broken_java.strip()}"


# --------------------------------------------------------------------------
# Compiler
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Diagnostic:
    kind: str  # "ERROR" | "WARNING" | ...
    line: int  # 1-based, -1 if unknown
    column: int  # 1-based, -1 if unknown
    code: str  # javac key, e.g. "compiler.err.cant.resolve.location"
    message: str  # human readable, single line


@dataclass(frozen=True)
class CompileResult:
    ok: bool
    diagnostics: tuple[Diagnostic, ...] = ()

    @property
    def errors(self) -> tuple[Diagnostic, ...]:
        return tuple(d for d in self.diagnostics if d.kind == "ERROR")

    def error_text(self, max_errors: int = 3) -> str:
        """Compact text fed back to the model in repair mode."""
        return " | ".join(f"line {d.line}: {d.message}" for d in self.errors[:max_errors])


@runtime_checkable
class JavaCompiler(Protocol):
    def compile(self, source: str, filename: str = "Main.java") -> CompileResult: ...

    def compile_many(self, sources: Sequence[str]) -> list[CompileResult]:
        """Compile independent sources (each as its own compilation unit)."""
        ...

    def close(self) -> None: ...


# --------------------------------------------------------------------------
# Generator (what the pipeline needs from the model)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 512
    num_beams: int = 1  # 1 = greedy
    do_sample: bool = False
    temperature: float = 1.0
    top_k: int = 0  # 0 = disabled
    num_return_sequences: int = 1


@runtime_checkable
class CodeGenerator(Protocol):
    def generate(self, sources: Sequence[str], config: GenerationConfig | None = None) -> list[list[str]]:
        """For each encoder source string return ``num_return_sequences``
        decoded Java strings, best first."""
        ...


# --------------------------------------------------------------------------
# Dataset records (one JSON object per line)
# --------------------------------------------------------------------------
@dataclass
class Example:
    id: str
    instruction: str  # natural-language shorthand
    java: str  # reference Java, guaranteed to compile
    spec: dict = field(default_factory=dict)  # structured description used for conformance checks


@dataclass
class RepairExample:
    id: str
    instruction: str
    broken_java: str  # does NOT compile
    error_text: str  # CompileResult.error_text() of broken_java
    java: str  # reference fix
