#!/usr/bin/env python
"""Run the whole pipeline in order: env check -> data -> tokenizer -> [lm] -> seq2seq -> RL -> evaluation ->
compare -> plots -> checkpoints/best -> mjavac demo.

    python scripts/run_all.py                      # full run with the gpu config (smoke pre-flight first if CUDA)
    python scripts/run_all.py --preset smoke       # seconds-long end-to-end check with the tiny config
    python scripts/run_all.py --only rl,eval       # selected stages        (--from STAGE, --skip a,b also exist)
    python scripts/run_all.py --force --only rl    # redo a finished stage from scratch

Stdlib only; every stage is a subprocess of the current interpreter, so run it with the interpreter of the
virtualenv that has torch installed.  Progress is kept in a small state file (runs/run_all_state.json); a finished
stage is skipped on re-run, an interrupted training stage continues with the trainer's own --resume.
Stage names: env data tokenizer lm seq2seq rl eval_retrieval eval_s2s eval_rl compare plots best demo
(aliases: train = lm,seq2seq,rl   eval = the three eval_* stages).  env, compare, plots, best, demo are cheap and
always re-run when selected.  lm only runs with --lm (always in the smoke preset).
"""
from __future__ import annotations

import argparse
import filecmp
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGES = ["env", "data", "tokenizer", "lm", "seq2seq", "rl", "eval_retrieval", "eval_s2s", "eval_rl",
          "compare", "plots", "best", "demo"]
ALIASES = {"train": ["lm", "seq2seq", "rl"], "eval": ["eval_retrieval", "eval_s2s", "eval_rl"]}
ALWAYS_RUN = {"env", "compare", "plots", "best", "demo"}
DOWNSTREAM = {  # when a stage re-runs, results computed from its output are stale
    "data": ["tokenizer", "lm", "seq2seq", "rl", "eval_retrieval", "eval_s2s", "eval_rl"],
    "tokenizer": ["lm", "seq2seq", "rl", "eval_s2s", "eval_rl"],
    "lm": ["seq2seq", "rl", "eval_s2s", "eval_rl"],
    "seq2seq": ["rl", "eval_s2s", "eval_rl"],
    "rl": ["eval_rl"],
}


class StageFailed(Exception):
    pass


# ----------------------------------------------------------------------------------------- helpers
def cmd_str(cmd: list[str]) -> str:
    return subprocess.list2cmdline(cmd) if os.name == "nt" else shlex.join(cmd)


def banner(msg: str) -> None:
    print(f"\n{'=' * 78}\n{msg}\n{'=' * 78}", flush=True)


def child_env() -> dict:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def run(cmd: list[str], tail: int = 60, quiet: bool = False) -> tuple[int, list[str]]:
    """Run ``cmd`` in the repo root, streaming its output live. Returns (exit code, last output lines)."""
    if not quiet:
        print(f"$ {cmd_str(cmd)}", flush=True)
    last: deque = deque(maxlen=tail)
    p = subprocess.Popen(cmd, cwd=ROOT, env=child_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace", bufsize=1)
    try:
        for line in p.stdout:  # type: ignore[union-attr]
            if not quiet:
                print(line, end="", flush=True)
            last.append(line)
        rc = p.wait()
    except KeyboardInterrupt:
        print("\n[run_all] Ctrl-C: waiting for the stage to save its state ...", flush=True)
        try:
            p.wait(timeout=180)
        except subprocess.TimeoutExpired:
            p.kill()
        raise
    return rc, list(last)


def py(*args: str) -> list[str]:
    return [sys.executable, *args]


class State:
    def __init__(self, path: Path):
        self.path = path
        try:
            self.data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.data = {}

    def done(self, stage: str) -> bool:
        return self.data.get(stage, {}).get("status") == "done"

    def set(self, stage: str, status: str, **extra) -> None:
        self.data[stage] = {"status": status, "at": time.strftime("%Y-%m-%d %H:%M:%S"), **extra}
        self.save()

    def drop(self, stages) -> list[str]:
        gone = [s for s in stages if s in self.data]
        for s in gone:
            del self.data[s]
        if gone:
            self.save()
        return gone

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        tmp.replace(self.path)


# ----------------------------------------------------------------------------------------- context
class Ctx:
    """All paths and settings derived from the command line."""

    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.smoke = a.preset == "smoke"
        self.config = "tiny" if self.smoke else a.config
        self.prefix = Path(self.config).stem
        if self.smoke:
            self.data = ROOT / "runs" / "smoke" / "data"
            self.ckpt = ROOT / "checkpoints" / "smoke"
            self.runs = ROOT / "runs" / "smoke"
        else:
            self.data = ROOT / a.data
            self.ckpt = ROOT / "checkpoints"
            self.runs = ROOT / "runs"
        self.tokenizer = self.ckpt / "tokenizer.json"
        self.best = self.ckpt / "best"
        self.lm_name, self.s2s_name, self.rl_name = (f"{self.prefix}_{s}" for s in ("lm", "s2s", "rl"))
        self.use_lm = a.lm or self.smoke
        self.cuda = False  # filled in by the env stage
        defaults = (0.2, 0.3, 0.3) if self.smoke else (30.0, 180.0, 90.0)
        given = (a.lm_minutes, a.s2s_minutes, a.rl_minutes)
        self.minutes = dict(zip(("lm", "seq2seq", "rl"), (g if g is not None else d for g, d in zip(given, defaults))))
        self.eval_limit = a.eval_limit or (40 if self.smoke else None)
        self.max_retries = 2 if self.smoke else a.max_retries

    def ckpt_dir(self, name: str) -> Path:
        return self.ckpt / name

    def common_train(self) -> list[str]:
        a = self.a
        out = ["--config", self.config, "--data", str(self.data), "--ckpt-root", str(self.ckpt),
               "--runs-root", str(self.runs), "--device", a.device]
        if a.no_amp:
            out.append("--no-amp")
        if a.threads:
            out += ["--threads", str(a.threads)]
        for s in a.set:
            out += ["--set", s]
        return out

    def smoke_sets(self) -> list[str]:
        # on a CUDA machine make the smoke run exercise the AMP code path of the supervised trainer too
        return ["--set", "train.amp=true"] if (self.smoke and self.cuda and not self.a.no_amp) else []


# ----------------------------------------------------------------------------------------- stages
ENV_PROBE = r"""
import json, sys
out = {"python": sys.version.split()[0]}
try:
    import torch
    out["torch"] = torch.__version__
    out["torch_cuda_build"] = torch.version.cuda
    out["cuda"] = bool(torch.cuda.is_available())
    if out["cuda"]:
        p = torch.cuda.get_device_properties(0)
        out.update(gpu=p.name, vram_gb=round(p.total_memory / 2**30, 1), n_gpus=torch.cuda.device_count(),
                   bf16=bool(torch.cuda.is_bf16_supported()))
except Exception as e:
    out["torch_error"] = repr(e)
for mod in ("tokenizers", "yaml", "numpy", "matplotlib", "mjava"):
    try:
        __import__(mod); out.setdefault("modules", {})[mod] = True
    except Exception as e:
        out.setdefault("modules", {})[mod] = False
print("PROBE " + json.dumps(out))
"""

JAVA_PROBE = r"""
from mjava.compiler import get_compiler
c = get_compiler("server")
ok = c.compile("class A { int f() { return 1; } }")
bad = c.compile("class A { int f() { return 1 } }")
print("JAVA", type(c).__name__, ok.ok, bad.ok)
c.close()
"""


def probe_env() -> tuple[int, dict]:
    rc, lines = run(py("-c", ENV_PROBE), quiet=True)
    info = {}
    for l in lines:
        if l.startswith("PROBE "):
            info = json.loads(l[6:])
    return rc, info


def stage_env(ctx: Ctx) -> None:
    a = ctx.a
    problems, warnings = [], []
    if sys.version_info < (3, 11):
        problems.append(f"Python {sys.version.split()[0]} is too old (need >= 3.11)")
    print(f"python      : {sys.version.split()[0]}  ({sys.executable})")
    rc, info = probe_env()
    if rc != 0 or not info:
        raise StageFailed("could not run the environment probe with this interpreter")
    if "torch_error" in info:
        problems.append(f"torch cannot be imported: {info['torch_error']}")
    else:
        print(f"torch       : {info['torch']} (built for CUDA {info['torch_cuda_build']})")
    for m, ok in info.get("modules", {}).items():
        if not ok:
            if m == "matplotlib":
                warnings.append("python module 'matplotlib' is missing: the plots stage will fail (pip install -e .[dev])")
            else:
                problems.append(f"python module '{m}' is missing (pip install -e .[dev] from the repo root)")
    ctx.cuda = bool(info.get("cuda"))
    if ctx.cuda:
        print(f"GPU         : {info['gpu']}  {info['vram_gb']} GiB  (bf16 {'yes' if info['bf16'] else 'no -> fp16 + GradScaler'}, "
              f"{info['n_gpus']} visible device(s); only device 0 is used)")
        if info["vram_gb"] < 8 and ctx.config == "gpu":  # gpu4 is the config meant for small cards
            warnings.append("less than 8 GiB of VRAM: gpu.yaml assumes 8-12 GiB. If CUDA runs out of memory use "
                            "--set train.batch_tokens=8000 --set rl.batch_examples=16 --set rl.num_samples=1 (see README)")
        if info["vram_gb"] >= 20 and ctx.config == "gpu":
            print("note        : >= 20 GiB VRAM: you can try --set train.batch_tokens=32000 for faster training")
    else:
        print("GPU         : none (torch.cuda.is_available() is False)")
        if a.device == "cuda":
            problems.append("--device cuda was given but CUDA is not available to torch (CPU-only torch wheel installed?)")
        elif ctx.config.startswith("gpu") and not a.allow_cpu and not ctx.smoke:
            problems.append(f"config '{ctx.config}' needs a CUDA GPU and none is visible. Install the CUDA build of torch "
                            "(README, 'Setup') or pass --allow-cpu if you really want to train on the CPU")
    # JDK, through the project's own compiler startup
    for tool in ("javac", "java"):
        if shutil.which(tool) is None:
            problems.append(f"'{tool}' not found on PATH: install a JDK 11 or newer")
    if shutil.which("javac"):
        try:
            r = subprocess.run(["javac", "-version"], capture_output=True, text=True, timeout=60)
            jv = (r.stdout + r.stderr).strip()
            print(f"JDK         : {jv}")
            digits = jv.split()[-1].split(".")
            major = int(digits[1]) if digits[0] == "1" else int(digits[0].split("-")[0])
            if major < 11:
                problems.append(f"{jv}: JDK 11 or newer required")
        except Exception as e:  # noqa: BLE001
            warnings.append(f"could not parse 'javac -version' ({e!r})")
        rc, lines = run(py("-c", JAVA_PROBE), quiet=True)
        res = [l.split() for l in lines if l.startswith("JAVA ")]
        if rc != 0 or not res:
            problems.append("the compile server did not start (see output above)")
        else:
            _, kind, good_ok, bad_ok = res[0]
            if good_ok != "True" or bad_ok != "False":
                problems.append(f"compiler self-test gave wrong answers (good compiles: {good_ok}, broken compiles: {bad_ok})")
            else:
                print(f"compiler    : {kind} self-test passed")
                if kind != "ServerCompiler":
                    warnings.append("fell back to the slow subprocess compiler; RL will be very slow")
    free = shutil.disk_usage(ROOT).free / 2**30
    print(f"free disk   : {free:.1f} GiB at {ROOT}")
    if free < 2:
        problems.append("less than 2 GiB free disk space")
    elif free < 5:
        warnings.append("less than 5 GiB free disk space (checkpoints with optimizer state are a few hundred MB each)")
    if not ctx.smoke and not (ctx.data / "meta.json").exists():
        print(f"data        : {ctx.data} will be built (needs the JDK; several minutes)")
    for w in warnings:
        print(f"WARNING: {w}")
    if problems:
        for p in problems:
            print(f"PROBLEM: {p}")
        raise StageFailed("environment check failed")
    print("environment OK")


def stage_data(ctx: Ctx) -> None:
    cmd = py("-m", "mjava.data.build", "--out", str(ctx.data))
    if ctx.smoke:
        cmd += ["--n-train", "300", "--n-val", "60", "--n-test", "60", "--n-repair-train", "100",
                "--n-repair-val", "30", "--n-repair-test", "30", "--workers", "2"]
    elif ctx.a.build_workers:
        cmd += ["--workers", str(ctx.a.build_workers)]
    do(cmd)


def stage_tokenizer(ctx: Ctx) -> None:
    do(py("-m", "mjava.tokenizer", "--data", str(ctx.data), "--out", str(ctx.tokenizer),
          "--vocab-size", "1500" if ctx.smoke else "6000"))


def _train_cmd(ctx: Ctx, module: str, stage_key: str, name: str, extra: list[str]) -> list[str]:
    cmd = py("-m", module, *extra, "--name", name, "--max-minutes", str(ctx.minutes[stage_key]), *ctx.common_train(),
             *ctx.smoke_sets())
    if not ctx.a.force and (ctx.ckpt_dir(name) / "last.pt").exists():
        cmd.append("--resume")
        print(f"[run_all] found {ctx.ckpt_dir(name) / 'last.pt'}: continuing that run with --resume "
              "(the time budget counts all sessions together)")
    return cmd


def stage_lm(ctx: Ctx) -> None:
    do(_train_cmd(ctx, "mjava.train.supervised", "lm", ctx.lm_name, ["--stage", "lm", "--tokenizer", str(ctx.tokenizer)]))


def stage_seq2seq(ctx: Ctx) -> None:
    extra = ["--stage", "seq2seq", "--tokenizer", str(ctx.tokenizer)]
    lm = ctx.ckpt_dir(ctx.lm_name)
    if ctx.use_lm and (lm / "model.pt").exists():
        extra += ["--init-from", str(lm)]
    do(_train_cmd(ctx, "mjava.train.supervised", "seq2seq", ctx.s2s_name, extra))


def stage_rl(ctx: Ctx) -> None:
    do(_train_cmd(ctx, "mjava.train.rl", "rl", ctx.rl_name, ["--init-from", str(ctx.ckpt_dir(ctx.s2s_name))]))


def _eval(ctx: Ctx, name: str, *who: str) -> None:
    cmd = py("-m", "mjava.eval.evaluate", *who, "--name", name, "--data", str(ctx.data), "--split", "test",
             "--runs-root", str(ctx.runs), "--max-retries", str(ctx.max_retries), "--device", ctx.a.device)
    if ctx.eval_limit:
        cmd += ["--limit", str(ctx.eval_limit)]
    do(cmd)


def stage_eval_retrieval(ctx: Ctx) -> None:
    _eval(ctx, "retrieval", "--baseline", "retrieval")


def stage_eval_s2s(ctx: Ctx) -> None:
    _eval(ctx, ctx.s2s_name, "--checkpoint", str(ctx.ckpt_dir(ctx.s2s_name)))


def stage_eval_rl(ctx: Ctx) -> None:
    _eval(ctx, ctx.rl_name, "--checkpoint", str(ctx.ckpt_dir(ctx.rl_name)))


def _eval_json(ctx: Ctx, name: str) -> Path:
    return ctx.runs / name / "eval.json"


def stage_compare(ctx: Ctx) -> None:
    files, labels = [], []
    for name, label in (("retrieval", "retrieval baseline"), (ctx.s2s_name, "supervised"), (ctx.rl_name, "supervised+RL")):
        if _eval_json(ctx, name).exists():
            files.append(str(_eval_json(ctx, name)))
            labels.append(label)
    if not files:
        raise StageFailed("no eval.json found; run the eval stages first")
    do(py("-m", "mjava.eval.compare", *files, "--labels", ",".join(labels), "--out", str(ctx.runs / "compare.md")))


def stage_plots(ctx: Ctx) -> None:
    names = [ctx.lm_name] if ctx.use_lm else []
    runs = [ctx.runs / n for n in names + [ctx.s2s_name, ctx.rl_name] if (ctx.runs / n / "metrics.jsonl").exists()]
    if not runs:
        raise StageFailed("no metrics.jsonl found; run the training stages first")
    cmd = py("-m", "mjava.eval.plots", "--out", str(ctx.runs / "plots"))
    for r in runs:
        cmd += ["--run", str(r)]
    do(cmd)


def _score(ev: dict) -> tuple:
    """Selection key: final compile-and-conform rate with the correction loop, then first-attempt equivalents."""
    c, f = ev.get("correction", {}), ev.get("first_attempt", {})
    sa = c.get("success_at", {})
    last = sa.get(str(max(map(int, sa)))) if sa else 0.0
    return (c.get("final_full_rate", 0.0), f.get("full_rate", 0.0), last, f.get("compile_rate", 0.0))


def stage_best(ctx: Ctx) -> None:
    cands = []
    for name, label in ((ctx.s2s_name, "supervised"), (ctx.rl_name, "supervised+RL")):
        p, d = _eval_json(ctx, name), ctx.ckpt_dir(name)
        if p.exists() and (d / "model.pt").exists():
            ev = json.loads(p.read_text(encoding="utf-8"))
            cands.append((label, name, d, ev, _score(ev)))
    if not cands:
        raise StageFailed("no evaluated checkpoint to choose from (eval_s2s / eval_rl missing)")
    print("selection (test set, with the self-correction loop): highest fraction that compiles AND matches the spec; "
          "ties go to first-attempt compile+conform, then success after the last retry, then first-attempt compile rate; "
          "a full tie keeps the supervised checkpoint")
    for label, name, d, ev, sc in cands:
        print(f"  {label:14s} {name:10s} final compile+conform {100 * sc[0]:5.1f}%  first-attempt compile+conform "
              f"{100 * sc[1]:5.1f}%  success@last-retry {100 * sc[2]:5.1f}%  first-attempt compile {100 * sc[3]:5.1f}%")
    win = cands[0]  # supervised comes first, so RL has to be strictly better
    identical = len(cands) == 2 and filecmp.cmp(cands[0][2] / "model.pt", cands[1][2] / "model.pt", shallow=False)
    if identical:
        print("note: the RL checkpoint is byte-identical to the supervised one (RL never beat the supervised model on "
              "its validation subset, so its best checkpoint is the starting point). Differences in the numbers above "
              "are sampling noise of the retry strategy; keeping the supervised checkpoint.")
    else:
        for c in cands[1:]:
            if c[4] > win[4]:
                win = c
    label, name, d, ev, sc = win
    if len(cands) == 2:
        print("RESULT: RL fine-tuning improved the selection metric; using supervised+RL." if name == ctx.rl_name else
              "RESULT: RL fine-tuning did NOT improve the selection metric; using the supervised checkpoint.")
    else:
        print(f"RESULT: only {label} was evaluated; using it.")
    if ctx.best.exists():
        shutil.rmtree(ctx.best)
    ctx.best.mkdir(parents=True)
    for f in ("model.pt", "config.yaml", "tokenizer.json"):
        shutil.copyfile(d / f, ctx.best / f)
    (ctx.best / "SOURCE.txt").write_text(
        f"copied from {d} ({label}) by scripts/run_all.py on {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"test final compile+conform {sc[0]:.4f}, first-attempt compile+conform {sc[1]:.4f}\n", encoding="utf-8")
    print(f"{ctx.best} <- {d} ({label})")


def stage_demo(ctx: Ctx) -> None:
    work = ctx.runs / "demo"
    work.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "examples" / "Student.mjava", work / "Student.mjava")  # copy: keeps examples/ clean
    for old in (work / "Student.java", work / "Student.mjava.review.txt"):
        old.unlink(missing_ok=True)
    cmd = py("-m", "mjava.cli", str(work / "Student.mjava"), "-o", str(work / "Student.java"), "--verbose")
    if ctx.smoke:
        cmd += ["--checkpoint", str(ctx.best)]
    elif os.environ.get("MJAVA_CHECKPOINT"):
        print(f"[run_all] note: MJAVA_CHECKPOINT={os.environ['MJAVA_CHECKPOINT']} is set and overrides checkpoints/best")
    rc, _ = run(cmd)
    if rc == 0:
        print(f"mjavac succeeded: {work / 'Student.java'}")
    elif rc == 1:
        print("mjavac exited 1: the generated Java still did not compile after the retries (normal for the tiny smoke "
              f"model; see {work / 'Student.mjava.review.txt'}). The correction loop itself ran.")
    else:
        raise StageFailed(f"mjavac crashed or hit an environment error (exit {rc}):\n  {cmd_str(cmd)}")


def do(cmd: list[str]) -> None:
    rc, tail = run(cmd)
    if rc != 0:
        hint = ""
        if "out of memory" in "".join(tail).lower():
            hint = ("\nCUDA ran out of memory. Lower the batch size and run this script again (finished stages are skipped, "
                    "an interrupted training stage resumes):\n  supervised: --set train.batch_tokens=8000 "
                    "(and/or --set train.max_batch_examples=64)\n  RL: --set rl.batch_examples=16 --set rl.num_samples=1 "
                    "(and/or --set rl.batch_tokens=8000)")
        raise StageFailed(f"command exited with code {rc}; re-run it by hand with:\n  {cmd_str(cmd)}{hint}")


FUNCS = {"data": stage_data, "tokenizer": stage_tokenizer, "lm": stage_lm, "seq2seq": stage_seq2seq,
         "rl": stage_rl, "eval_retrieval": stage_eval_retrieval, "eval_s2s": stage_eval_s2s, "eval_rl": stage_eval_rl,
         "compare": stage_compare, "plots": stage_plots, "best": stage_best, "demo": stage_demo}


def outputs_exist(ctx: Ctx, stage: str) -> bool:
    """A 'done' state is only trusted if the stage's main output is still on disk."""
    c = ctx
    return {
        "tokenizer": lambda: c.tokenizer.exists(),
        "lm": lambda: (c.ckpt_dir(c.lm_name) / "model.pt").exists(),
        "seq2seq": lambda: (c.ckpt_dir(c.s2s_name) / "model.pt").exists(),
        "rl": lambda: (c.ckpt_dir(c.rl_name) / "model.pt").exists(),
        "eval_retrieval": lambda: _eval_json(c, "retrieval").exists(),
        "eval_s2s": lambda: _eval_json(c, c.s2s_name).exists(),
        "eval_rl": lambda: _eval_json(c, c.rl_name).exists(),
    }.get(stage, lambda: True)()


# ----------------------------------------------------------------------------------------- main
def select(a: argparse.Namespace) -> list[str]:
    def expand(csv: str) -> list[str]:
        out: list[str] = []
        for s in filter(None, (x.strip() for x in csv.split(","))):
            out += ALIASES.get(s, [s])
        bad = [s for s in out if s not in STAGES]
        if bad:
            raise SystemExit(f"unknown stage(s) {bad}; known: {', '.join(STAGES)} (aliases: {', '.join(ALIASES)})")
        return out

    sel = list(STAGES)
    if a.only:
        want = expand(a.only)
        sel = [s for s in STAGES if s in want]
    if a.from_stage:
        start = expand(a.from_stage)[0]
        sel = [s for s in sel if STAGES.index(s) >= STAGES.index(start)]
    if a.skip:
        skip = expand(a.skip)
        sel = [s for s in sel if s not in skip]
    return sel


def run_pipeline(ctx: Ctx, sel: list[str]) -> list[tuple[str, str, float]]:
    state = State(ctx.runs / ("state.json" if ctx.smoke else "run_all_state.json"))
    report: list[tuple[str, str, float]] = []
    for stage in sel:
        banner(f"[{stage}]  ({time.strftime('%H:%M:%S')})")
        t0 = time.time()
        if stage == "data" and (ctx.data / "meta.json").exists() and not ctx.a.force:
            print(f"{ctx.data / 'meta.json'} exists: dataset build skipped (use --force to rebuild)")
            report.append((stage, "skipped (exists)", 0.0))
            continue
        if stage not in ALWAYS_RUN and stage != "data" and state.done(stage) and not ctx.a.force and outputs_exist(ctx, stage):
            print(f"already done ({state.data[stage]['at']}); skipped. Use --force to redo it.")
            report.append((stage, "skipped (done)", 0.0))
            continue
        if stage == "lm" and not ctx.use_lm:
            print("lm pretraining not enabled (pass --lm to run it); skipping")
            report.append((stage, "not enabled", 0.0))
            continue
        if stage == "eval_rl" and not (ctx.ckpt_dir(ctx.rl_name) / "model.pt").exists():
            print("no RL checkpoint; skipping")
            report.append((stage, "no rl checkpoint", 0.0))
            continue
        if stage not in ALWAYS_RUN:
            state.set(stage, "running")
        try:
            FUNCS[stage](ctx)
        except StageFailed as e:
            state.set(stage, "failed", error=str(e)[:500])
            report.append((stage, "FAILED", time.time() - t0))
            print(f"\n[run_all] stage '{stage}' FAILED: {e}", flush=True)
            raise
        except KeyboardInterrupt:
            state.set(stage, "interrupted")
            report.append((stage, "interrupted", time.time() - t0))
            raise
        dt = time.time() - t0
        if stage not in ALWAYS_RUN:
            state.set(stage, "done", seconds=round(dt, 1))
            stale = state.drop(DOWNSTREAM.get(stage, []))
            # a downstream trainer must restart from the new upstream output, not --resume its old run
            names = {"lm": ctx.lm_name, "seq2seq": ctx.s2s_name, "rl": ctx.rl_name}
            for down in DOWNSTREAM.get(stage, []):
                if down in names:
                    (ctx.ckpt_dir(names[down]) / "last.pt").unlink(missing_ok=True)
            if stale:
                print(f"[run_all] '{stage}' re-ran, so these earlier results are stale and will be redone: {', '.join(stale)}")
        report.append((stage, "ok", dt))
        print(f"[run_all] stage '{stage}' finished in {dt:.1f}s", flush=True)
    return report


def print_report(report, total: float) -> None:
    banner("summary")
    for s, status, dt in report:
        print(f"  {s:18s} {status:18s} {dt:8.1f}s")
    print(f"  {'total':18s} {'':18s} {total:8.1f}s")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="gpu", help="tiny|cpu|gpu|gpu4 (4 GB card) or a yaml path (default gpu)")
    ap.add_argument("--preset", choices=["full", "smoke"], default="full",
                    help="smoke: tiny config, a tiny temporary dataset, seconds-long budgets, separate output dirs "
                         "(runs/smoke, checkpoints/smoke); never touches data/, checkpoints/<config>_* or runs/<config>_*")
    ap.add_argument("--only", help="comma separated stages (or aliases train, eval)")
    ap.add_argument("--from", dest="from_stage", help="start at this stage")
    ap.add_argument("--skip", help="comma separated stages to leave out")
    ap.add_argument("--force", action="store_true", help="redo selected stages even if finished (training restarts from scratch)")
    ap.add_argument("--data", default="data", help="dataset directory (default data)")
    ap.add_argument("--lm", action="store_true", help="also run the optional lm pretraining stage before seq2seq")
    ap.add_argument("--lm-minutes", type=float, default=None, help="time budget of the lm stage (default 30)")
    ap.add_argument("--s2s-minutes", type=float, default=None, help="time budget of the seq2seq stage (default 180)")
    ap.add_argument("--rl-minutes", type=float, default=None, help="time budget of the RL stage (default 90)")
    ap.add_argument("--eval-limit", type=int, default=None, help="evaluate only the first N test examples (default: all)")
    ap.add_argument("--max-retries", type=int, default=3, help="self-correction retries in the evaluation (default 3)")
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda, forwarded to the trainers and the evaluation")
    ap.add_argument("--no-amp", action="store_true", help="disable mixed precision on CUDA")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads (forwarded)")
    ap.add_argument("--set", action="append", default=[], metavar="K=V",
                    help="config override forwarded to the trainers, e.g. train.batch_tokens=8000 or rl.lr=2e-5 (repeatable)")
    ap.add_argument("--build-workers", type=int, default=None, help="compile worker processes of the data build")
    ap.add_argument("--allow-cpu", action="store_true", help="allow the gpu config on a machine without CUDA")
    ap.add_argument("--skip-smoke", action="store_true", help="skip the smoke pre-flight before a full run on a CUDA machine")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    ctx = Ctx(a)
    sel = select(a)
    t_all = time.time()
    banner(f"run_all  preset={a.preset} config={ctx.config} stages={','.join(sel)}")
    report: list = []
    try:
        if "env" in sel:
            t0 = time.time()
            banner(f"[env]  ({time.strftime('%H:%M:%S')})")
            try:
                stage_env(ctx)
            except StageFailed as e:
                print(f"\n[run_all] environment check FAILED: {e}")
                return 1
            report.append(("env", "ok", time.time() - t0))
            sel = [s for s in sel if s != "env"]
        elif ctx.smoke:
            ctx.cuda = bool(probe_env()[1].get("cuda"))
        heavy = [s for s in sel if s not in ALWAYS_RUN]
        if ctx.cuda and not ctx.smoke and heavy and not a.skip_smoke:
            banner("smoke pre-flight on this GPU (tiny model, short): a broken CUDA/AMP/JDK path should fail here")
            t0 = time.time()
            cmd = py(str(Path(__file__).resolve()), "--preset", "smoke", "--force", "--device", a.device)
            if a.no_amp:
                cmd.append("--no-amp")
            rc, _ = run(cmd)
            if rc != 0:
                print(f"\n[run_all] smoke pre-flight FAILED; the real run was not started. Debug with:\n  {cmd_str(cmd)}")
                return 1
            report.append(("smoke pre-flight", "ok", time.time() - t0))
            banner("smoke pre-flight passed; starting the real run")
        report += run_pipeline(ctx, sel)
    except StageFailed:
        print_report(report, time.time() - t_all)
        print("\n[run_all] pipeline stopped. Fix the problem, then run the same command again: finished stages are "
              "skipped and an interrupted training stage resumes.")
        return 1
    except KeyboardInterrupt:
        print("\n[run_all] interrupted; state saved. Run the same command again to continue.")
        return 130
    print_report(report, time.time() - t_all)
    if ctx.best.exists():
        print(f"\nBest checkpoint: {ctx.best}" + ("" if ctx.smoke else "   (mjavac uses checkpoints/best by default)"))
    if (ctx.runs / "compare.md").exists():
        print(f"Results table  : {ctx.runs / 'compare.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
