"""``.mjava`` parsing: block detection and file assembly.

Block detection rules
---------------------
A ``.mjava`` file is split into *chunks* at blank lines, then each chunk into
blocks.  Line endings (LF, CRLF, CR) are normalised to ``\\n`` for processing.

1. Blank lines separate chunks, except that a blank line does NOT end a chunk
   when it is inside a ``/* ... */`` comment, or when the chunk starts like a
   Java declaration (rule 4) and its braces are still unbalanced.  So a
   hand-written class with blank lines between its methods stays one block.
2. Leading comment lines of a chunk (``// ...`` and ``/* ... */``, including
   javadoc) become a ``comment`` block, kept verbatim.  Comments stay attached
   (no blank line inserted) to the block that follows them in the same chunk.
   The slide example (two ``//`` lines directly above the sentence) therefore
   yields one comment block and one shorthand block.
3. ``@gen`` forces shorthand: a remaining chunk whose first line starts with
   ``@gen`` is shorthand and the text after the marker (plus following lines)
   is the instruction.
4. ``java`` (passed through unchanged) if the remaining chunk
   a. is a ``package`` / ``import`` declaration (``package a.b;`` /
      ``import a.b.*;`` / ``import static a.B.c;``), or
   b. starts with an annotation, a modifier (``public private protected
      static final abstract sealed non-sealed strictfp synchronized
      default``) or ``class interface enum record`` (lower case, so the
      sentence ``Class Student has ...`` is not caught) AND contains ``{`` or
      ends with ``;``, or
   c. ends with ``}`` or ``;``, has balanced braces, and contains at least one
      of ``( ) { } = [ ]`` (so a plain sentence that happens to end with ``;``
      is still shorthand).
   Text like ``public class Foo with a name field`` fails (b) and is shorthand.
5. Everything else is ``shorthand``: its non-comment lines are stripped and
   joined with single spaces into one instruction.  Comment lines inside a
   shorthand chunk are dropped from the instruction.

Assembly
--------
``assemble`` joins the blocks again.  Comment blocks at the very start of the
file stay on top; then the ``package`` line (moved up from wherever it was; the
first one wins, so a ``package`` is always the first code line), then all ``import`` lines
(hand-written or generated) are hoisted right after it and de-duplicated, and
the remaining body of each block follows in the original order.  Every output
line records which block and which line of that block's own text it came from
so compiler diagnostics can be mapped back to a block.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Mapping, Sequence

from mjava.interfaces import CompileResult, Diagnostic

COMMENT, JAVA, SHORTHAND = "comment", "java", "shorthand"

_MODIFIERS = (
    "public|private|protected|static|final|abstract|sealed|non-sealed|strictfp|"
    "synchronized|default|class|interface|enum|record"
)
_JAVA_START = re.compile(rf"^(?:@(?!gen\b)\w[\w.]*|(?:{_MODIFIERS})\b)")
_GEN_MARKER = re.compile(r"^@gen\b[:\s]*(.*)$", re.DOTALL)
_PACKAGE = re.compile(r"^\s*package\s+[\w.]+\s*;?\s*$")
_IMPORT = re.compile(r"^\s*import\s+(?:static\s+)?[\w.]+(?:\.\*)?\s*;?\s*$")
_JAVA_CHARS = re.compile(r"[(){}=\[\]]")
_TYPE_DECL = re.compile(
    r"^\s*(?:(?:public|protected|private|abstract|final|static|sealed|non-sealed|strictfp)\s+)*"
    r"(class|interface|enum|record)\s+([A-Za-z_]\w*)",
    re.MULTILINE,
)
_PUBLIC_TYPE = re.compile(
    r"^\s*(?:(?:protected|private|abstract|final|static|sealed|non-sealed|strictfp)\s+)*public\s+"
    r"(?:(?:abstract|final|static|sealed|non-sealed|strictfp)\s+)*(?:class|interface|enum|record)\s+([A-Za-z_]\w*)",
    re.MULTILINE,
)


@dataclass
class Block:
    index: int
    kind: str  # "comment" | "java" | "shorthand"
    text: str  # original text of the block, lines joined with "\n"
    start_line: int  # 1-based line in the (normalised) input file
    end_line: int
    instruction: str | None = None  # shorthand only
    attached: bool = False  # comment block directly above the next block (no blank line)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "kind": self.kind,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "instruction": self.instruction,
        }


# --------------------------------------------------------------------------
# low level scanning helpers
# --------------------------------------------------------------------------
def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def detect_newline(text: str) -> str:
    crlf = text.count("\r\n")
    lf = text.count("\n") - crlf
    return "\r\n" if crlf > lf else "\n"


def brace_depth(text: str) -> int:
    """Net brace depth, ignoring braces in strings, chars and comments."""
    depth = 0
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        two = text[i : i + 2]
        if two == "//":
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif two == "/*":
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
        elif c == '"' and text[i : i + 3] == '"""':
            j = text.find('"""', i + 3)
            i = n if j < 0 else j + 3
        elif c in "\"'":
            i += 1
            while i < n and text[i] != c and text[i] != "\n":
                i += 2 if text[i] == "\\" else 1
            i += 1
        else:
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            i += 1
    return depth


def _split_leading_comments(lines: list[str]) -> int:
    """Number of leading lines that are pure comment lines."""
    i, n = 0, len(lines)
    while i < n:
        s = lines[i].strip()
        if s.startswith("//"):
            i += 1
        elif s.startswith("/*"):
            j = i
            seg = s[2:]
            while "*/" not in seg:
                j += 1
                if j >= n:
                    return n  # unterminated comment swallows the rest
                seg = lines[j]
            if seg.split("*/", 1)[1].strip():
                return i  # code after the closing */: not a pure comment
            i = j + 1
        else:
            break
    return i


def _is_java_chunk(lines: list[str]) -> bool:
    text = "\n".join(lines).strip()
    if not text:
        return False
    first = lines[0].strip()
    if _GEN_MARKER.match(first):
        return False
    if len(lines) == 1 and (_PACKAGE.match(first) or _IMPORT.match(first)) and (
        first.endswith(";") or "." in first
    ):
        return True
    if _JAVA_START.match(first) and ("{" in text or text.endswith(";")):
        return True
    if text[-1] in "};" and brace_depth(text) == 0 and _JAVA_CHARS.search(text):
        return True
    return False


def _starts_java_decl(lines: list[str]) -> bool:
    k = _split_leading_comments(lines)
    rest = lines[k:]
    return bool(rest) and bool(_JAVA_START.match(rest[0].strip())) and not _GEN_MARKER.match(rest[0].strip())


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------
def _chunks(lines: list[str]) -> list[tuple[int, list[str]]]:
    """Group lines into chunks separated by blank lines -> (start_line, lines)."""
    chunks: list[tuple[int, list[str]]] = []
    cur: list[str] = []
    start = 0
    in_block_comment = False
    for no, line in enumerate(lines, 1):
        if not line.strip():
            if cur and (
                in_block_comment
                or (_starts_java_decl(cur) and brace_depth("\n".join(cur)) > 0)
            ):
                cur.append(line)
                continue
            if cur:
                while cur and not cur[-1].strip():
                    cur.pop()
                chunks.append((start, cur))
                cur = []
            continue
        if not cur:
            start = no
        cur.append(line)
        # track /* */ state
        s = line
        pos = 0
        while True:
            if in_block_comment:
                j = s.find("*/", pos)
                if j < 0:
                    break
                in_block_comment = False
                pos = j + 2
            else:
                j = s.find("/*", pos)
                k = s.find("//", pos)
                if j < 0 or (0 <= k < j):
                    break
                in_block_comment = True
                pos = j + 2
    if cur:
        while cur and not cur[-1].strip():
            cur.pop()
        chunks.append((start, cur))
    return chunks


def parse_mjava(text: str) -> list[Block]:
    """Split ``text`` into :class:`Block` objects (see module docstring)."""
    lines = normalize_newlines(text).split("\n")
    blocks: list[Block] = []

    def add(kind, blines, start, instruction=None, attached=False):
        blocks.append(
            Block(len(blocks), kind, "\n".join(blines), start, start + len(blines) - 1, instruction, attached)
        )

    for start, chunk in _chunks(lines):
        k = _split_leading_comments(chunk)
        if k:
            add(COMMENT, chunk[:k], start, attached=k < len(chunk))
        rest = chunk[k:]
        if not rest:
            continue
        rstart = start + k
        m = _GEN_MARKER.match(rest[0].strip())
        if m:
            parts = [m.group(1).strip()] + [l.strip() for l in rest[1:]]
            add(SHORTHAND, rest, rstart, " ".join(p for p in parts if p))
        elif _is_java_chunk(rest):
            add(JAVA, rest, rstart)
        else:
            parts = [l.strip() for l in rest if l.strip() and not _is_comment_line(l)]
            add(SHORTHAND, rest, rstart, " ".join(parts))
    return blocks


def _is_comment_line(line: str) -> bool:
    s = line.strip()
    return s.startswith("//") or (s.startswith("/*") and s.endswith("*/")) or s.startswith("*")


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------
def split_header(code: str) -> tuple[list[tuple[str, int]], list[tuple[str, int]], list[tuple[str, int]]]:
    """Split Java source into (packages, imports, body) line lists.

    Each entry is ``(line_text, 1-based line number in code)``.  Package and
    import lines are only recognised before the first declaration.
    """
    packages: list[tuple[str, int]] = []
    imports: list[tuple[str, int]] = []
    body: list[tuple[str, int]] = []
    in_header = True
    in_comment = False
    for no, line in enumerate(code.split("\n"), 1):
        if in_header:
            s = line.strip()
            if in_comment:
                body.append((line, no))
                if "*/" in s:
                    in_comment = False
                continue
            if not s or s.startswith("//"):
                body.append((line, no))
                continue
            if s.startswith("/*"):
                body.append((line, no))
                in_comment = "*/" not in s
                continue
            if _PACKAGE.match(line):
                packages.append((line.strip().rstrip(";") + ";", no))
                continue
            if _IMPORT.match(line):
                imports.append((line.strip().rstrip(";") + ";", no))
                continue
            in_header = False
        body.append((line, no))
    return packages, imports, body


def public_type_name(code: str) -> str | None:
    m = _PUBLIC_TYPE.search(code)
    return m.group(1) if m else None


def first_type_name(code: str) -> str | None:
    m = _TYPE_DECL.search(code)
    return m.group(2) if m else None


def filename_for(code: str, default: str = "Main.java") -> str:
    """File name javac needs for ``code`` (matches a public top-level type)."""
    name = public_type_name(code)
    return f"{name}.java" if name else default


@dataclass
class Assembled:
    text: str
    # per output line (0-based list index = line number - 1): (block index | None, line in that block's code, 1-based)
    origin: list[tuple[int | None, int]] = field(default_factory=list)

    def locate(self, line: int) -> tuple[int | None, int]:
        if 1 <= line <= len(self.origin):
            return self.origin[line - 1]
        return (None, -1)

    def split_result(self, result: CompileResult) -> dict[int | None, list[Diagnostic]]:
        """Group diagnostics by owning block, lines rewritten to block-relative
        (1-based line within the block's own code, -1 if unknown)."""
        out: dict[int | None, list[Diagnostic]] = {}
        for d in result.diagnostics:
            owner, raw = self.locate(d.line) if d.line > 0 else (None, -1)
            out.setdefault(owner, []).append(replace(d, line=raw if owner is not None else d.line))
        return out


def assemble(blocks: Sequence[Block], code_for: Mapping[int, str | None]) -> Assembled:
    """Assemble the output Java file.

    ``code_for`` maps shorthand block index -> generated Java.  Shorthand
    blocks missing from the mapping (or mapped to ``None``) are omitted.
    """
    package: tuple[str, int, int] | None = None  # text, owner, raw line
    imports: dict[str, tuple[int, int]] = {}
    bodies: list[tuple[Block, list[tuple[str, int]]]] = []
    for b in blocks:
        if b.kind == COMMENT:
            bodies.append((b, [(l, i) for i, l in enumerate(b.text.split("\n"), 1)]))
            continue
        code = b.text if b.kind == JAVA else code_for.get(b.index)
        if code is None:
            continue
        pk, im, body = split_header(code)
        if pk and package is None:
            package = (pk[0][0], b.index, pk[0][1])
        for text, no in im:
            imports.setdefault(text, (b.index, no))
        while body and not body[0][0].strip() and b.kind != COMMENT:
            body.pop(0)
        while body and not body[-1][0].strip():
            body.pop()
        if body:
            bodies.append((b, body))

    out: list[str] = []
    origin: list[tuple[int | None, int]] = []

    def emit(text: str, owner: int | None, raw: int) -> None:
        out.append(text)
        origin.append((owner, raw))

    # leading comments (file header) stay on top, before package/imports
    lead = 0
    while lead < len(bodies) and bodies[lead][0].kind == COMMENT:
        lead += 1

    def emit_blocks(items) -> None:
        for pos, (b, body) in enumerate(items):
            for text, no in body:
                emit(text, b.index, no)
            if pos + 1 < len(items) and not (b.kind == COMMENT and b.attached):
                emit("", None, -1)

    if package or imports:
        emit_blocks(bodies[:lead])
        if lead:
            emit("", None, -1)
        if package:
            emit(package[0], package[1], package[2])
            if imports:
                emit("", None, -1)
        for text, (owner, raw) in imports.items():
            emit(text, owner, raw)
        if bodies[lead:]:
            emit("", None, -1)
        emit_blocks(bodies[lead:])
    else:
        emit_blocks(bodies)
    text = "\n".join(out)
    if text:
        text += "\n"
    return Assembled(text, origin)
