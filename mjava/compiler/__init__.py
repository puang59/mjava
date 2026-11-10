"""Java compile services."""
from __future__ import annotations

from mjava.compiler.subprocess_compiler import SubprocessCompiler


def get_compiler(kind: str = "auto"):
    """kind: "auto" (server, falling back to subprocess), "server" or "subprocess"."""
    if kind == "subprocess":
        return SubprocessCompiler()
    from mjava.compiler.server import ServerCompiler

    if kind == "server":
        return ServerCompiler()
    try:
        return ServerCompiler()
    except Exception:
        return SubprocessCompiler()


__all__ = ["get_compiler", "SubprocessCompiler"]
