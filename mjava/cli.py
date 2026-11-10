"""``mjavac``: expand a ``.mjava`` file into compilable Java.

    mjavac File.mjava [-o File.java] [--checkpoint DIR] [--max-retries N]
                      [--keep-temp] [--verbose] [--run]

Exit codes
    0  success: the ``.java`` file was written
    1  the generated Java still did not compile after the retry limit; nothing
       was written except ``File.mjava.review.txt``
    2  usage or environment error (missing input, checkpoint, compiler or
       model code, unexpected crash)
    3  ``--run`` only: the program compiled but exited non-zero (or could not run)

The input file is only ever opened for reading.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

from mjava.pipeline.correct import SelfCorrector
from mjava.pipeline.mjava_file import first_type_name, public_type_name
from mjava.pipeline.report import format_attempt, format_review, format_summary

DEFAULT_CHECKPOINT = os.environ.get("MJAVA_CHECKPOINT", "checkpoints/best")


class CliError(Exception):
    """Environment/usage problem -> exit code 2."""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mjavac", description="Expand .mjava shorthand into compilable Java.")
    p.add_argument("input", help="the .mjava file (never modified)")
    p.add_argument("-o", "--output", help="output .java path (default: next to the input)")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                   help="model checkpoint directory (default: %(default)s, or $MJAVA_CHECKPOINT)")
    p.add_argument("--max-retries", type=int, default=3, help="retries per block after the first attempt (default 3)")
    p.add_argument("--keep-temp", action="store_true", help="keep the temp directory with the expansion")
    p.add_argument("--verbose", "-v", action="store_true", help="show every attempt and the error fed back")
    p.add_argument("--run", action="store_true", help="after a successful compile, run the program's main()")
    return p


def _load_compiler():
    try:
        from mjava.compiler import get_compiler
    except ImportError as e:
        raise CliError(f"the Java compiler service (mjava.compiler) is not available: {e}")
    try:
        return get_compiler()
    except Exception as e:  # noqa: BLE001
        raise CliError(f"could not start a Java compiler (is a JDK on PATH?): {e}")


def _load_generator(checkpoint: str):
    path = Path(checkpoint)
    if not path.is_dir():
        raise CliError(
            f"checkpoint directory not found: {checkpoint}\n"
            "Train a model first (see README) or pass --checkpoint DIR / set MJAVA_CHECKPOINT."
        )
    try:
        from mjava.model.generator import TransformerGenerator
    except ImportError as e:
        raise CliError(f"the model code (mjava.model.generator) could not be imported: {e}")
    try:
        return TransformerGenerator.from_checkpoint(str(path))
    except Exception as e:  # noqa: BLE001
        raise CliError(f"could not load the checkpoint {checkpoint}: {e}")


def _run_program(java: str, tmp: Path, stem: str) -> int:
    pub = public_type_name(java)
    src = tmp / "run" / ((pub or stem) + ".java")
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(java, encoding="utf-8", newline="")
    pkg = re.search(r"^\s*package\s+([\w.]+)\s*;", java, re.MULTILINE)
    mains = [m.start() for m in re.finditer(r"static\s+void\s+main\s*\(", java)]
    cls = None
    if mains:
        decls = [m for m in re.finditer(r"\b(?:class|enum|record|interface)\s+([A-Za-z_]\w*)", java)
                 if m.start() < mains[0]]
        cls = decls[-1].group(1) if decls else first_type_name(java)
    if cls is None:
        print("mjavac: --run: no main method found", file=sys.stderr)
        return 3
    outdir = tmp / "run" / "out"
    try:
        r = subprocess.run(["javac", "-d", str(outdir), str(src)], capture_output=True, text=True)
        if r.returncode:
            print(r.stderr, file=sys.stderr)
            return 3
        qual = f"{pkg.group(1)}.{cls}" if pkg else cls
        print(f"--- running {qual} ---")
        sys.stdout.flush()
        r = subprocess.run(["java", "-cp", str(outdir), qual])
        return 0 if r.returncode == 0 else 3
    except FileNotFoundError as e:
        print(f"mjavac: --run needs a JDK on PATH: {e}", file=sys.stderr)
        return 3


def main(argv=None, *, generator=None, compiler=None) -> int:
    """Entry point.  ``generator`` / ``compiler`` can be injected (tests)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:  # noqa: BLE001
            pass
    args = build_parser().parse_args(argv)
    inp = Path(args.input)
    owns_compiler = False
    tmp: Path | None = None
    try:
        if not inp.is_file():
            raise CliError(f"input file not found: {inp}")
        if args.max_retries < 0:
            raise CliError("--max-retries must be >= 0")
        if compiler is None:
            compiler = _load_compiler()
            owns_compiler = True
        if generator is None:
            generator = _load_generator(args.checkpoint)

        def on_attempt(trace, attempt):
            if args.verbose:
                print(format_attempt(trace, attempt), flush=True)

        corrector = SelfCorrector(generator, compiler, max_retries=args.max_retries, on_attempt=on_attempt)
        result = corrector.run_file(inp)

        tmp = Path(tempfile.mkdtemp(prefix="mjavac-"))
        out = Path(args.output) if args.output else inp.with_suffix(".java")
        tmp_java = tmp / out.name
        tmp_java.write_text(result.java, encoding="utf-8", newline="")
        review = inp.with_name(inp.name + ".review.txt")

        if not result.blocks and result.ok:
            print(f"mjavac: note: no shorthand blocks found in {inp}; Java passed through unchanged",
                  file=sys.stderr)
        if args.verbose and result.blocks:
            print(format_summary(result))

        if result.ok:
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(tmp_java, out)
            if review.exists():
                review.unlink()
            pub = public_type_name(result.java)
            if pub and pub != out.stem:
                print(f"mjavac: warning: public type {pub} should be in {pub}.java, not {out.name}",
                      file=sys.stderr)
            print(f"mjavac: wrote {out}")
            if args.run:
                return _run_program(result.java, tmp, out.stem)
            return 0

        review.write_text(format_review(result, inp.name), encoding="utf-8")
        print(f"mjavac: FAILED: {inp} still does not compile after {args.max_retries} retries", file=sys.stderr)
        for d in result.file_diagnostics:
            if d.kind == "ERROR":
                print(f"  line {d.line}: {d.message}", file=sys.stderr)
        print(f"mjavac: flagged for manual review, see {review}", file=sys.stderr)
        return 1
    except CliError as e:
        print(f"mjavac: error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("mjavac: interrupted", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001
        print(f"mjavac: unexpected error: {type(e).__name__}: {e}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 2
    finally:
        if owns_compiler and compiler is not None:
            try:
                compiler.close()
            except Exception:  # noqa: BLE001
                pass
        if tmp is not None:
            if args.keep_temp:
                print(f"mjavac: temp files kept in {tmp}", file=sys.stderr)
            else:
                shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
