"""Evaluate a checkpoint (or the retrieval baseline) on a dataset split.

    python -m mjava.eval.evaluate --checkpoint checkpoints/gpu_s2s --data data --split test --name gpu_s2s --max-retries 3
    python -m mjava.eval.evaluate --checkpoint checkpoints/gpu_rl  --data data --split test --name gpu_rl
    python -m mjava.eval.evaluate --baseline retrieval --data data --split test --name retrieval

Writes ``runs/<name>/eval.json``, ``eval.md`` and ``eval_examples.jsonl`` (one record per example).

eval.json
    schema, name, system ("transformer" | "retrieval_baseline"), checkpoint, data, split, limit, n, max_retries, device,
    seconds, timestamp, notes[]
    perplexity: {loss, ppl, tokens, ppl_gen, ppl_fix}            teacher-forced on split (+ repair_<split>); null for baseline
    first_attempt: {compile_rate, exact_match, mean_conformance, full_rate}
        compile_rate     attempt 0 compiles; exact_match = equal to the reference modulo whitespace;
        mean_conformance spec_conformance(spec, attempt 0); full_rate = compiles AND conformance == 1
    correction: (the self-correction loop, SelfCorrector.correct_instruction, `max_retries` retries)
        success_at: {"0": .., "1": .., ..}   fraction whose final code compiles after <= k retries (non-decreasing in k)
        full_success_at: same but also fully conformant (a loop that converges to compilable-but-wrong code shows here)
        mean_attempts, final_mean_conformance, final_full_rate, final_exact_match,
        converged_nonconformant_rate   compiles at the end but conformance < 1
    by_family: {family: {n, first_attempt, correction}}          spec["family"]; counts sum to n
    heldout:   {seen: {n, ...}, heldout: {n, ...}}               spec["heldout"] (identifiers only present in test)
    repair: {n, compile_rate, exact_match, mean_conformance, full_rate, n_with_spec}   repair_<split>.jsonl, greedy
The baseline has no repair ability: its "correction" block repeats the first attempt (see notes).
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Optional, Sequence

from mjava.data.conformance import required_elements, spec_conformance
from mjava.eval.baseline import RetrievalBaseline
from mjava.eval.compare import render_comparison
from mjava.interfaces import GenerationConfig, build_generate_source, build_repair_source
from mjava.train.common import evaluate_loss, load_pairs, pick_device, read_jsonl


def norm(s: str) -> str:
    return " ".join((s or "").split())


def _mean(xs: Sequence) -> float:
    return float(sum(xs)) / len(xs) if xs else 0.0


def conformance(spec: Optional[dict], java: Optional[str]) -> float:
    if not java or not spec or not required_elements(spec):
        return 0.0
    try:
        return spec_conformance(spec, java)
    except Exception:
        return 0.0


def summarize(recs: Sequence[dict], max_retries: int) -> dict:
    """Aggregate per-example records (see ``eval_examples.jsonl``) into the metric blocks of eval.json."""
    n = len(recs)
    first = {
        "compile_rate": _mean([r["a0_ok"] for r in recs]),
        "exact_match": _mean([r["a0_exact"] for r in recs]),
        "mean_conformance": _mean([r["a0_conf"] for r in recs]),
        "full_rate": _mean([r["a0_ok"] and r["a0_conf"] >= 1 - 1e-9 for r in recs]),
    }
    succ, full = {}, {}
    for k in range(max_retries + 1):
        ok_k = [r["succeeded_at"] is not None and r["succeeded_at"] <= k for r in recs]
        succ[str(k)] = _mean(ok_k)
        full[str(k)] = _mean([o and r["final_conf"] >= 1 - 1e-9 for o, r in zip(ok_k, recs)])
    corr = {
        "success_at": succ,
        "full_success_at": full,
        "mean_attempts": _mean([r["attempts"] for r in recs]),
        "final_mean_conformance": _mean([r["final_conf"] for r in recs]),
        "final_full_rate": _mean([r["final_ok"] and r["final_conf"] >= 1 - 1e-9 for r in recs]),
        "final_exact_match": _mean([r["final_exact"] for r in recs]),
        "converged_nonconformant_rate": _mean([r["final_ok"] and r["final_conf"] < 1 - 1e-9 for r in recs]),
    }
    return {"n": n, "first_attempt": first, "correction": corr}


def breakdowns(recs: Sequence[dict], max_retries: int) -> dict:
    out = summarize(recs, max_retries)
    fams = sorted({r["family"] for r in recs})
    out["by_family"] = {f: summarize([r for r in recs if r["family"] == f], max_retries) for f in fams}
    out["heldout"] = {
        key: summarize(sub, max_retries)
        for key, sub in (("seen", [r for r in recs if not r["heldout"]]), ("heldout", [r for r in recs if r["heldout"]]))
        if sub
    }
    return out


class CachedGenerator:
    """Wraps a CodeGenerator; greedy outputs prefetched in one big batch are served to the one-at-a-time
    SelfCorrector calls, everything else goes to the wrapped generator."""

    def __init__(self, inner):
        self.inner = inner
        self.cache: dict = {}
        self.hits = 0

    @staticmethod
    def _key(src: str, cfg: GenerationConfig):
        return (src, cfg.max_new_tokens, cfg.num_beams, cfg.do_sample, cfg.temperature, cfg.top_k, cfg.num_return_sequences)

    def prefetch(self, sources: Sequence[str], cfg: GenerationConfig) -> None:
        outs = self.inner.generate(list(sources), cfg)
        for s, o in zip(sources, outs):
            self.cache[self._key(s, cfg)] = o

    def generate(self, sources, config=None):
        cfg = config or GenerationConfig()
        res = [None] * len(sources)
        miss = []
        for i, s in enumerate(sources):
            hit = self.cache.get(self._key(s, cfg))
            if hit is not None and not cfg.do_sample:
                res[i] = list(hit)
                self.hits += 1
            else:
                miss.append(i)
        if miss:
            outs = self.inner.generate([sources[i] for i in miss], cfg)
            for i, o in zip(miss, outs):
                res[i] = o
        return res


def _example_record(ex: dict, a0_code: Optional[str], a0_ok: bool, final_code: Optional[str], final_ok: bool,
                    succeeded_at: Optional[int], attempts: int) -> dict:
    spec = ex.get("spec") or {}
    ref = norm(ex["java"])
    return {
        "id": ex.get("id"), "family": spec.get("family", "unknown"), "heldout": bool(spec.get("heldout", False)),
        "instruction": ex["instruction"], "a0_ok": bool(a0_ok), "a0_conf": conformance(spec, a0_code),
        "a0_exact": norm(a0_code or "") == ref, "final_ok": bool(final_ok), "final_conf": conformance(spec, final_code),
        "final_exact": norm(final_code or "") == ref, "succeeded_at": succeeded_at, "attempts": attempts,
        "a0_code": a0_code, "final_code": final_code,
    }


def eval_transformer(generator, compiler, examples: list[dict], max_retries: int, log=print) -> list[dict]:
    from mjava.pipeline.correct import SelfCorrector

    cg = CachedGenerator(generator)
    srcs = [build_generate_source(norm(e["instruction"])) for e in examples]
    t0 = time.time()
    cg.prefetch(srcs, GenerationConfig(max_new_tokens=512))
    log(f"  attempt-0 generation for {len(srcs)} examples: {time.time() - t0:.1f}s")
    sc = SelfCorrector(cg, compiler, max_retries=max_retries)
    recs = []
    for i, ex in enumerate(examples):
        try:
            tr = sc.correct_instruction(ex["instruction"])
            a0 = tr.attempts[0]
            recs.append(_example_record(ex, a0.code, a0.ok, tr.code, tr.success, tr.succeeded_at, len(tr.attempts)))
        except Exception as e:  # noqa: BLE001 - one odd example must not lose the whole evaluation
            log(f"  example {ex.get('id')} crashed the loop: {e!r}; counted as failure")
            recs.append(_example_record(ex, None, False, None, False, None, 1))
        if (i + 1) % 50 == 0:
            log(f"  corrected {i + 1}/{len(examples)} ({time.time() - t0:.0f}s)")
    return recs


def eval_retrieval(base: RetrievalBaseline, compiler, examples: list[dict]) -> list[dict]:
    outs = [base.generate_one(e["instruction"]) for e in examples]
    results = compiler.compile_many(outs)
    return [_example_record(e, o, r.ok, o, r.ok, 0 if r.ok else None, 1) for e, o, r in zip(examples, outs, results)]


def eval_repair(generator, compiler, data_dir: Path, split: str, limit: Optional[int], spec_by_id: dict) -> Optional[dict]:
    p = data_dir / f"repair_{split}.jsonl"
    if not p.exists():
        return None
    recs = read_jsonl(p)[:limit]
    if not recs:
        return None
    srcs = [build_repair_source(r["instruction"], r.get("error_text", ""), r["broken_java"]) for r in recs]
    outs = [o[0] if o else "" for o in generator.generate(srcs, GenerationConfig(max_new_tokens=512))]
    res = compiler.compile_many(outs)
    ok = [x.ok for x in res]
    specs = [spec_by_id.get(r["id"][len("repair-"):] if r["id"].startswith("repair-") else r["id"]) for r in recs]
    conf = [conformance(sp, o) for sp, o in zip(specs, outs)]
    has = [bool(sp) and bool(required_elements(sp)) for sp in specs]
    return {
        "n": len(recs),
        "compile_rate": _mean(ok),
        "exact_match": _mean([norm(o) == norm(r["java"]) for o, r in zip(outs, recs)]),
        "n_with_spec": sum(has),
        "mean_conformance": _mean([c for c, h in zip(conf, has) if h]),
        "full_rate": _mean([o and c >= 1 - 1e-9 for o, c, h in zip(ok, conf, has) if h]),
    }


def check_inputs(checkpoint: Optional[str], data_dir: Path, split: str) -> None:
    problems = []
    if checkpoint:
        for f in ("model.pt", "config.yaml", "tokenizer.json"):
            if not (Path(checkpoint) / f).exists():
                problems.append(f"{Path(checkpoint) / f} not found")
    if not (data_dir / f"{split}.jsonl").exists():
        problems.append(f"{data_dir / (split + '.jsonl')} not found (python -m mjava.data.build --out {data_dir})")
    if shutil.which("javac") is None:
        problems.append("javac not found on PATH: install a JDK (11 or newer)")
    if problems:
        raise SystemExit("evaluation start-up check failed:\n  - " + "\n  - ".join(problems))


def run_eval(
    *, name: str, data_dir: str = "data", split: str = "test", checkpoint: Optional[str] = None, baseline: Optional[str] = None,
    limit: Optional[int] = None, max_retries: int = 3, device: str = "auto", batch_size: int = 32,
    runs_root: str = "runs", compiler=None, compiler_kind: str = "auto", log=print,
) -> dict:
    if bool(checkpoint) == bool(baseline):
        raise SystemExit("give exactly one of --checkpoint DIR or --baseline retrieval")
    if baseline not in (None, "retrieval"):
        raise SystemExit("--baseline must be 'retrieval'")
    d = Path(data_dir)
    check_inputs(checkpoint, d, split)
    t0 = time.time()
    own_compiler = compiler is None
    if own_compiler:
        from mjava.compiler import get_compiler

        compiler = get_compiler(compiler_kind)
    examples = read_jsonl(d / f"{split}.jsonl")[:limit]
    if not examples:
        raise SystemExit(f"{d / (split + '.jsonl')} is empty")
    notes: list[str] = []
    n_nospec = sum(1 for e in examples if not (e.get("spec") and required_elements(e["spec"])))
    if n_nospec:
        notes.append(f"{n_nospec}/{len(examples)} examples have no checkable spec: conformance counted as 0 for them")
    spec_by_id = {e["id"]: e.get("spec") for e in examples}
    used_dev = device
    try:
        if checkpoint:
            from mjava.model import TransformerGenerator

            dev = used_dev = pick_device(device)
            gen = TransformerGenerator.from_checkpoint(checkpoint, dev, batch_size)
            mc = gen.model.cfg
            log(f"evaluating {checkpoint} on {len(examples)} {split} examples (device={dev})")
            ppl = None
            try:
                g, f = load_pairs(gen.tokenizer, d, split, mc.max_src_len, mc.max_tgt_len, limit, limit)
                allp = g + f
                r = evaluate_loss(gen.model, allp, dev) if allp else {"loss": float("nan"), "ppl": float("nan"), "tokens": 0}
                ppl = {"loss": r["loss"], "ppl": r["ppl"], "tokens": r["tokens"],
                       "ppl_gen": evaluate_loss(gen.model, g, dev)["ppl"] if g else None,
                       "ppl_fix": evaluate_loss(gen.model, f, dev)["ppl"] if f else None}
                log(f"  perplexity {ppl['ppl']:.3f} (gen {ppl['ppl_gen']}, repair {ppl['ppl_fix']})")
            except Exception as e:  # noqa: BLE001
                notes.append(f"perplexity failed: {e!r}")
            recs = eval_transformer(gen, compiler, examples, max_retries, log)
            repair = eval_repair(gen, compiler, d, split, limit, spec_by_id)
            system = "transformer"
        else:
            train = read_jsonl(d / "train.jsonl")
            log(f"retrieval baseline: {len(train)} train examples -> {len(examples)} {split} examples")
            recs = eval_retrieval(RetrievalBaseline(train), compiler, examples)
            ppl, repair, system = None, None, "retrieval_baseline"
            notes.append("no repair ability: the correction block repeats the first attempt; perplexity and repair are n/a")
    finally:
        if own_compiler:
            try:
                compiler.close()
            except Exception:
                pass
    res = {
        "schema": 1, "name": name, "system": system, "checkpoint": checkpoint, "data": str(data_dir), "split": split,
        "limit": limit, "max_retries": max_retries, "device": used_dev, "seconds": round(time.time() - t0, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "notes": notes, "perplexity": ppl,
        **breakdowns(recs, max_retries), "repair": repair,
    }
    out = Path(runs_root) / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (out / "eval.md").write_text(f"# Evaluation: {name} ({split}, n={res['n']})\n\n" + render_comparison([res], [name]), encoding="utf-8")
    with open(out / "eval_examples.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    log(f"wrote {out / 'eval.json'}")
    return res


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None, help="checkpoint dir (model.pt, config.yaml, tokenizer.json)")
    ap.add_argument("--baseline", default=None, choices=["retrieval"], help="evaluate the non-neural baseline instead")
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="test")
    ap.add_argument("--name", required=True)
    ap.add_argument("--limit", type=int, default=None, help="only the first N examples (and N repair examples)")
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--compiler", default="auto", choices=["auto", "server", "subprocess"])
    ap.add_argument("--runs-root", default="runs")
    a = ap.parse_args(argv)
    run_eval(name=a.name, data_dir=a.data, split=a.split, checkpoint=a.checkpoint, baseline=a.baseline, limit=a.limit,
             max_retries=a.max_retries, device=a.device, batch_size=a.batch_size, runs_root=a.runs_root,
             compiler_kind=a.compiler)


if __name__ == "__main__":
    main()
