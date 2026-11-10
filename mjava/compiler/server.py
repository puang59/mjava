"""ServerCompiler: talks to one persistent JVM running CompileServer."""
from __future__ import annotations

import atexit
import json
import queue
import shutil
import subprocess
import threading
import weakref
from pathlib import Path
from typing import Sequence

from mjava.compiler.common import derive_filename, result_from_json
from mjava.interfaces import CompileResult, Diagnostic

_JAVA_DIR = Path(__file__).resolve().parent / "java"
_SRC = _JAVA_DIR / "CompileServer.java"
_BUILD = _JAVA_DIR / "build"
_build_lock = threading.Lock()
_live: "weakref.WeakSet[ServerCompiler]" = weakref.WeakSet()


def _close_all() -> None:
    for c in list(_live):
        try:
            c.close()
        except Exception:
            pass


atexit.register(_close_all)


def build_server() -> Path:
    """Compile CompileServer.java into build/ (if missing or stale). Returns the class dir."""
    cls = _BUILD / "CompileServer.class"
    with _build_lock:
        if cls.exists() and cls.stat().st_mtime >= _SRC.stat().st_mtime:
            return _BUILD
        javac = shutil.which("javac")
        if not javac:
            raise RuntimeError("javac not found: install a JDK (11 or newer) and put javac on PATH")
        _BUILD.mkdir(parents=True, exist_ok=True)
        p = subprocess.run(
            [javac, "-encoding", "UTF-8", "-d", str(_BUILD), str(_SRC)],
            capture_output=True, text=True,
        )
        if p.returncode != 0:
            raise RuntimeError("failed to build CompileServer: " + p.stderr)
    return _BUILD


class ServerCompiler:
    """JavaCompiler backed by a persistent in-memory javac JVM. Thread-safe."""

    def __init__(self, timeout: float = 20.0):
        self.timeout = timeout
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._q: "queue.Queue[str | None]" = queue.Queue()
        self._next_id = 0
        self.restarts = 0
        self._closed = False
        _live.add(self)
        with self._lock:
            self._start()
        r = self.compile("class Probe {}")  # fail fast if the server cannot answer
        if not r.ok:
            self.close()
            raise RuntimeError("compile server self-test failed: %r" % (r,))

    # -- process management -------------------------------------------------
    def _start(self) -> None:
        build = build_server()
        java = shutil.which("java")
        if not java:
            raise RuntimeError("java not found: install a JDK (11 or newer) and put java on PATH")
        self._q = q = queue.Queue()
        proc = subprocess.Popen(
            [java, "-Xshare:auto", "-Dfile.encoding=UTF-8", "-cp", str(build), "CompileServer"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        self._proc = proc

        def reader() -> None:
            try:
                for raw in proc.stdout:  # type: ignore[union-attr]
                    q.put(raw.decode("utf-8", errors="replace"))
            except Exception:
                pass
            q.put(None)

        threading.Thread(target=reader, daemon=True).start()

    def _kill(self) -> None:
        p, self._proc = self._proc, None
        if p is None:
            return
        try:
            if p.stdin:
                p.stdin.close()
        except Exception:
            pass
        try:
            p.kill()
        except Exception:
            pass
        try:
            p.wait(timeout=5)
        except Exception:
            pass

    def close(self) -> None:
        self._closed = True
        with self._lock:
            self._kill()

    def __enter__(self) -> "ServerCompiler":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self._kill()
        except Exception:
            pass

    # -- protocol ---------------------------------------------------------------
    def _roundtrip(self, req_id: int, payload: bytes) -> dict | None:
        p = self._proc
        if p is None or p.poll() is not None:
            return None
        try:
            p.stdin.write(payload)  # type: ignore[union-attr]
            p.stdin.flush()  # type: ignore[union-attr]
        except Exception:
            return None
        while True:
            try:
                line = self._q.get(timeout=self.timeout)
            except queue.Empty:
                return None
            if line is None:
                return None
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if obj.get("id") == req_id:
                return obj

    def compile(self, source: str, filename: str | None = None) -> CompileResult:
        if filename is None or filename == "Main.java":
            filename = derive_filename(source, filename or "Main.java")
        with self._lock:
            if self._closed:
                raise RuntimeError("compiler is closed")
            self._next_id += 1
            rid = self._next_id
            payload = (json.dumps({"id": rid, "filename": filename, "source": source}, ensure_ascii=True) + "\n").encode("ascii")
            obj = None
            for _ in range(2):
                if self._proc is None or self._proc.poll() is not None:
                    self._kill()
                    self._start()
                    self.restarts += 1
                obj = self._roundtrip(rid, payload)
                if obj is not None:
                    break
                self._kill()  # hung or died: restart and retry once
            if obj is None:
                return CompileResult(False, (Diagnostic("ERROR", -1, -1, "server.timeout",
                                                        "compile server did not respond"),))
            return result_from_json(obj)

    def compile_many(self, sources: Sequence[str]) -> list[CompileResult]:
        return [self.compile(s) for s in sources]
