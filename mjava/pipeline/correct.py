"""Self-correction loop.

Programmatic API
----------------
::

    from mjava.pipeline import SelfCorrector
    sc = SelfCorrector(generator, compiler, max_retries=3)
    result = sc.run(mjava_text)            # whole .mjava text -> CorrectionResult
    result = sc.run_file("Student.mjava")  # reads the file in binary mode only
    trace  = sc.correct_instruction("Create a class A with field x as int.")  # one block (evaluation)
    json.dumps(result.to_dict())           # the trace is plain JSON

``generator`` is a ``CodeGenerator`` and ``compiler`` a ``JavaCompiler``
(``mjava.interfaces``).  Nothing here touches the file system.

Algorithm (per shorthand block; ``max_retries`` = retries after attempt 0, so
at most ``1 + max_retries`` attempts per block, in total across both phases)
-------------------------------------------------------------------------
* attempt 0, strategy ``generate``: encoder source ``build_generate_source``;
  all blocks are generated in one batched call.
* attempt 1, 2 (strategy ``repair``): ``build_repair_source(instruction,
  error_text, previous_code)`` built from the newest failure, greedy.
* attempt >= 3 (strategy ``sample``): same repair source, ``num_samples``
  sampled candidates; each untried candidate is compiled and the first that
  compiles is taken (otherwise the first untried one).
* A candidate identical (modulo whitespace) to one already tried for the block
  is never used again.  If a ``repair`` attempt yields only tried candidates it
  escalates to sampling within the same attempt (``escalated_from`` records
  this); if sampling yields nothing new too, the attempt is recorded with
  ``code=None``, the block is marked exhausted and no further attempts happen.

Phase 1 ("block"): each block is compiled together with the hand-written Java
blocks of the file (but not with other shorthand blocks).  Only diagnostics
that map to the block's own lines count.  If the only errors are unresolved
symbols and the file has several shorthand blocks, the block is *deferred*
(the symbol may be defined by a sibling) rather than retried.

Phase 2 ("file"): the assembled file is compiled.  Each diagnostic is mapped
to the block owning that output line; only those blocks are retried (one
attempt per round), then the file is recompiled.  Errors that fall in
hand-written Java or have no location cannot be repaired and end the loop.
The loop stops when the file compiles, no block is retryable any more, or all
budgets are used, so it always terminates.

Trace shape (``CorrectionResult.to_dict()``)
--------------------------------------------
::

    {"ok": bool, "max_retries": int, "java": str,        # assembled file (last attempt of every block)
     "error_text": str,                                  # remaining file-level errors (empty when ok)
     "file_diagnostics": [Diagnostic dict, ...],         # from the last whole-file compile
     "overview": [{"index","kind","start_line","end_line","instruction"}, ...],  # every block
     "blocks": [{                                        # shorthand blocks only
         "index", "start_line", "end_line", "instruction",
         "success": bool, "succeeded_at": int | None,    # attempt index that produced the final compiling code
         "retries_used": int,                            # len(attempts) - 1
         "attempts": [{
             "index": int, "strategy": "generate"|"repair"|"sample",
             "phase": "block"|"file", "source": str,      # exact encoder source fed to the model
             "config": {GenerationConfig fields}, "code": str | None,
             "ok": bool, "diagnostics": [Diagnostic dict], "error_text": str,
             "num_candidates": int, "skipped_duplicates": int,
             "escalated_from": str | None, "note": str}]}]}

Diagnostic dicts are ``{kind,line,column,code,message}`` with ``line`` relative
to the block's own generated code (1-based), -1 if unknown.  "Success after k
retries" for a block = ``success and retries_used <= k`` or, to ask it of any
k from one trace, ``succeeded_at is not None and succeeded_at <= k``.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, replace
from typing import Callable, Sequence

from mjava.interfaces import (
    CodeGenerator,
    CompileResult,
    Diagnostic,
    GenerationConfig,
    JavaCompiler,
    build_generate_source,
    build_repair_source,
)
from mjava.pipeline.mjava_file import (
    COMMENT,
    JAVA,
    SHORTHAND,
    Assembled,
    Block,
    assemble,
    filename_for,
    parse_mjava,
)

STRATEGY_GENERATE, STRATEGY_REPAIR, STRATEGY_SAMPLE = "generate", "repair", "sample"
EMPTY_CODE = "mjava.empty"


def _diag_dict(d: Diagnostic) -> dict:
    return asdict(d)


def _norm(code: str) -> str:
    return " ".join(code.split())


def _error_text(diags: Sequence[Diagnostic], max_errors: int = 3) -> str:
    return CompileResult(False, tuple(diags)).error_text(max_errors)


@dataclass
class Attempt:
    index: int
    strategy: str
    phase: str  # "block" | "file"
    source: str
    config: dict
    code: str | None
    ok: bool
    diagnostics: list[dict] = field(default_factory=list)
    error_text: str = ""
    num_candidates: int = 0
    skipped_duplicates: int = 0
    escalated_from: str | None = None
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BlockTrace:
    index: int
    start_line: int
    end_line: int
    instruction: str
    attempts: list[Attempt] = field(default_factory=list)

    def last_coded(self) -> Attempt | None:
        """Newest attempt that produced code (the one whose code is in the output)."""
        for a in reversed(self.attempts):
            if a.code is not None:
                return a
        return None

    @property
    def success(self) -> bool:
        a = self.last_coded()
        return a is not None and a.ok

    @property
    def succeeded_at(self) -> int | None:
        return self.last_coded().index if self.success else None

    @property
    def retries_used(self) -> int:
        return max(0, len(self.attempts) - 1)

    @property
    def code(self) -> str | None:
        """Newest generated code (None if nothing was ever generated)."""
        for a in reversed(self.attempts):
            if a.code is not None:
                return a.code
        return None

    def success_within(self, k: int) -> bool:
        return self.succeeded_at is not None and self.succeeded_at <= k

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "instruction": self.instruction,
            "success": self.success,
            "succeeded_at": self.succeeded_at,
            "retries_used": self.retries_used,
            "attempts": [a.to_dict() for a in self.attempts],
        }


@dataclass
class CorrectionResult:
    ok: bool
    java: str
    blocks: list[BlockTrace]
    overview: list[dict]
    max_retries: int
    file_diagnostics: list[Diagnostic] = field(default_factory=list)
    newline: str = "\n"

    @property
    def error_text(self) -> str:
        return "" if self.ok else _error_text(self.file_diagnostics, 5)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "max_retries": self.max_retries,
            "java": self.java,
            "error_text": self.error_text,
            "file_diagnostics": [_diag_dict(d) for d in self.file_diagnostics],
            "overview": self.overview,
            "blocks": [b.to_dict() for b in self.blocks],
        }


@dataclass
class _State:
    block: Block
    trace: BlockTrace
    tried: set = field(default_factory=set)
    code: str | None = None
    diags: tuple = ()  # newest diagnostics, block-relative
    exhausted: bool = False
    deferred: bool = False

    @property
    def retries_left_used(self) -> int:
        return len(self.trace.attempts) - 1


class SelfCorrector:
    def __init__(
        self,
        generator: CodeGenerator,
        compiler: JavaCompiler,
        max_retries: int = 3,
        *,
        num_samples: int = 4,
        sample_temperature: float = 0.8,
        sample_top_k: int = 40,
        max_new_tokens: int = 512,
        on_attempt: Callable[[BlockTrace, Attempt], None] | None = None,
    ):
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.generator = generator
        self.compiler = compiler
        self.max_retries = max_retries
        self.num_samples = max(1, num_samples)
        self.sample_temperature = sample_temperature
        self.sample_top_k = sample_top_k
        self.max_new_tokens = max_new_tokens
        self.on_attempt = on_attempt

    # ------------------------------------------------------------------ API
    def run_file(self, path: str | os.PathLike) -> CorrectionResult:
        """Read ``path`` (binary, read-only) and run the loop."""
        with open(path, "rb") as f:
            raw = f.read()
        return self.run(raw.decode("utf-8-sig"))

    def run(self, text: str) -> CorrectionResult:
        from mjava.pipeline.mjava_file import detect_newline

        blocks = parse_mjava(text)
        overview = [b.to_dict() for b in blocks]
        newline = detect_newline(text)
        shorthand = [b for b in blocks if b.kind == SHORTHAND]
        states = {
            b.index: _State(b, BlockTrace(b.index, b.start_line, b.end_line, b.instruction or ""))
            for b in shorthand
        }

        if not shorthand:
            return self._no_shorthand(blocks, overview, newline)

        # attempt 0: one batched generate call for all blocks
        sources = [build_generate_source(b.instruction or "") for b in shorthand]
        cfg = GenerationConfig(max_new_tokens=self.max_new_tokens)
        outs = self._generate(sources, cfg)
        multi = len(shorthand) > 1
        for b, src, cands in zip(shorthand, sources, outs):
            st = states[b.index]
            self._first_attempt(blocks, st, src, cfg, cands)
        # phase 1
        for b in shorthand:
            st = states[b.index]
            while not self._block_done(st):
                if multi and self._only_unresolved(st):
                    st.deferred = True
                    break
                self._advance(blocks, states, st, phase="block")

        # phase 2
        final = self._file_phase(blocks, states)
        ok, asm, res = final
        traces = [states[b.index].trace for b in shorthand]
        diags = [] if ok else list(res.diagnostics)
        java = asm.text.replace("\n", newline) if newline != "\n" else asm.text
        return CorrectionResult(ok, java, traces, overview, self.max_retries, diags, newline)

    def correct_instruction(self, instruction: str) -> BlockTrace:
        """Run the loop for a single instruction (no file around it).  Used by
        the evaluation to measure first-attempt and success-after-k-retries."""
        res = self.run("@gen " + " ".join(instruction.split()))
        return res.blocks[0]

    # ------------------------------------------------------------ internals
    def _no_shorthand(self, blocks, overview, newline) -> CorrectionResult:
        asm = assemble(blocks, {})
        java = asm.text.replace("\n", newline) if newline != "\n" else asm.text
        if not any(b.kind == JAVA for b in blocks):
            return CorrectionResult(True, java, [], overview, self.max_retries, [], newline)
        res = self.compiler.compile(asm.text, filename_for(asm.text))
        return CorrectionResult(res.ok, java, [], overview, self.max_retries, list(res.diagnostics), newline)

    def _generate(self, sources: list[str], cfg: GenerationConfig) -> list[list[str]]:
        outs = self.generator.generate(sources, cfg)
        outs = [list(o) for o in outs]
        while len(outs) < len(sources):
            outs.append([])
        return outs

    @staticmethod
    def _clean(code: str) -> str:
        code = (code or "").replace("\r\n", "\n").strip("\n")
        s = code.strip()
        if s.startswith("```"):
            lines = s.split("\n")[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            code = "\n".join(lines)
        return code.strip()

    def _budget_left(self, st: _State) -> bool:
        return st.retries_left_used < self.max_retries

    def _block_done(self, st: _State) -> bool:
        a = st.trace.attempts[-1]
        return (a.ok and a.code is not None) or st.exhausted or not self._budget_left(st)

    @staticmethod
    def _only_unresolved(st: _State) -> bool:
        errs = [d for d in st.diags if d.kind == "ERROR"]
        return bool(errs) and all(
            "cant.resolve" in d.code or "cannot find symbol" in d.message.lower() for d in errs
        )

    def _check(self, blocks, states, st: _State, code: str, phase: str) -> tuple[bool, tuple[Diagnostic, ...]]:
        """Compile ``code`` in the block's context; return (ok, block-relative diagnostics)."""
        if not code.strip():
            return False, (Diagnostic("ERROR", -1, -1, EMPTY_CODE, "the model produced no code"),)
        if phase == "block":
            code_for = {st.block.index: code}
        else:
            code_for = {i: s.code for i, s in states.items() if s.code is not None}
            code_for[st.block.index] = code
        asm = assemble(blocks, code_for)
        res = self.compiler.compile(asm.text, filename_for(asm.text))
        return self._owned(asm, res, st.block.index)

    @staticmethod
    def _owned(asm: Assembled, res: CompileResult, idx: int) -> tuple[bool, tuple[Diagnostic, ...]]:
        groups = asm.split_result(res)
        mine = tuple(groups.get(idx, ()))
        errors = [d for d in mine if d.kind == "ERROR"]
        return (not errors), mine

    def _record(self, st: _State, att: Attempt) -> None:
        st.trace.attempts.append(att)
        if att.code is not None:
            st.code = att.code
            st.tried.add(_norm(att.code))
        st.diags = tuple(Diagnostic(**d) for d in att.diagnostics)
        if self.on_attempt:
            self.on_attempt(st.trace, att)

    def _first_attempt(self, blocks, st: _State, src: str, cfg: GenerationConfig, cands: list[str]) -> None:
        code = self._clean(cands[0]) if cands else ""
        ok, diags = self._check(blocks, {}, st, code, "block")
        self._record(
            st,
            Attempt(
                0, STRATEGY_GENERATE, "block", src, asdict(cfg), code, ok,
                [_diag_dict(d) for d in diags], _error_text(diags), len(cands), 0, None,
                "" if cands else "generator returned no candidate",
            ),
        )

    def _advance(self, blocks, states, st: _State, phase: str) -> None:
        """Make exactly one more attempt for the block (must have budget)."""
        n = len(st.trace.attempts)
        prev = st.code or ""
        err = _error_text(st.diags)
        src = build_repair_source(st.block.instruction or "", err, prev)
        strategy = STRATEGY_REPAIR if n < 3 else STRATEGY_SAMPLE
        escalated = None
        skipped = 0
        candidates: list[str] = []
        total = 0

        def fresh(raw: list[str]) -> list[str]:
            nonlocal skipped
            out, seen = [], set(st.tried)
            for r in raw:
                c = self._clean(r)
                k = _norm(c)
                if not c or k in seen:
                    skipped += 1
                    continue
                seen.add(k)
                out.append(c)
            return out

        cfg = self._config(strategy)
        raw = self._generate([src], cfg)[0]
        total += len(raw)
        candidates = fresh(raw)
        if not candidates and strategy == STRATEGY_REPAIR:
            escalated, strategy = STRATEGY_REPAIR, STRATEGY_SAMPLE
            cfg = self._config(strategy)
            raw = self._generate([src], cfg)[0]
            total += len(raw)
            candidates = fresh(raw)
        if not candidates:
            st.exhausted = True
            self._record(
                st,
                Attempt(n, strategy, phase, src, asdict(cfg), None, False, [d for d in map(_diag_dict, st.diags)],
                        err, total, skipped, escalated, "no untried candidate; block exhausted"),
            )
            return

        chosen, chosen_diags, chosen_ok = None, (), False
        for c in candidates if strategy == STRATEGY_SAMPLE else candidates[:1]:
            ok, diags = self._check(blocks, states, st, c, phase)
            if chosen is None or ok:
                chosen, chosen_diags, chosen_ok = c, diags, ok
            st.tried.add(_norm(c))
            if ok:
                break
        self._record(
            st,
            Attempt(n, strategy, phase, src, asdict(cfg), chosen, chosen_ok,
                    [_diag_dict(d) for d in chosen_diags], _error_text(chosen_diags), total, skipped, escalated, ""),
        )

    def _config(self, strategy: str) -> GenerationConfig:
        if strategy == STRATEGY_SAMPLE:
            return GenerationConfig(
                max_new_tokens=self.max_new_tokens, do_sample=True, temperature=self.sample_temperature,
                top_k=self.sample_top_k, num_return_sequences=self.num_samples,
            )
        return GenerationConfig(max_new_tokens=self.max_new_tokens)

    def _file_phase(self, blocks, states) -> tuple[bool, Assembled, CompileResult]:
        while True:
            code_for = {i: s.code for i, s in states.items() if s.code is not None}
            asm = assemble(blocks, code_for)
            res = self.compiler.compile(asm.text, filename_for(asm.text))
            groups = asm.split_result(res)
            if res.ok:
                for st in states.values():
                    st.diags = ()
                    a = st.trace.last_coded()
                    if a is not None and not a.ok:
                        a.ok, a.diagnostics, a.error_text = True, [], ""
                        a.note = (a.note + " resolved at file level").strip()
                return True, asm, res
            failing = []
            for i, st in states.items():
                mine = groups.get(i, [])
                errs = [d for d in mine if d.kind == "ERROR"]
                a = st.trace.last_coded()
                if a is None:
                    continue
                a.ok = not errs
                if errs:
                    a.diagnostics = [_diag_dict(d) for d in mine]
                    a.error_text = _error_text(mine)
                st.diags = tuple(mine)
                if errs:
                    failing.append(st)
            fixable = [s for s in failing if self._budget_left(s) and not s.exhausted]
            if not fixable:
                return False, asm, res
            for st in fixable:
                self._advance(blocks, states, st, phase="file")
