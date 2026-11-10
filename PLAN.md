# Implementation Plan — Self-Correcting Java Code Generation (Transformers + RL)

Team: Aayush Shukla (23BPS1081), Karan Kumar (23BPS1034), Aryan Kumar (23BPS1193)

## 1. What we are building

A command-line compiler wrapper, `mjavac`, that takes a `.mjava` file containing
natural-language shorthand ("Create a class Student with fields: name as String,
age as int, gpa as double.") and produces a compilable `.java` file.

1. The wrapper scans the file and finds the shorthand blocks.
2. A Transformer (trained by us, from scratch) expands each block into Java.
3. The expansion is written to a temp file; the original is never modified.
4. The temp file is compiled. On failure the compiler error is fed back to the
   model, which produces a corrected version.
5. This repeats until it compiles or the retry limit is hit, in which case the
   file is flagged for manual review.

Training has three stages: supervised next-token training (cross-entropy,
evaluated by perplexity on held-out code), supervised repair training
(broken code + compiler error -> fixed code), then RL fine-tuning where the
reward comes from the Java compiler.

## 2. Constraints that shape the design

| Fact | Consequence |
|---|---|
| Dev machine: i5-13500H (12 cores), 16 GB RAM, **no CUDA GPU** | **No training runs on this machine** (user decision). Only seconds-long `tiny`-config unit tests run here. Inference and the `mjavac` CLI must still work here on CPU. |
| Training happens on a separate PC with a CUDA GPU | The `gpu` config is the real one. Training code auto-detects the device, uses mixed precision on CUDA, and saves checkpoints that load on CPU. Data build, compile service and RL must be OS-independent (the GPU machine may be Linux / another JDK). CUDA paths cannot be tested here and are delivered unverified. |
| Python 3.14 venv at `.venv/` (torch CPU, tokenizers, pytest, ... installed there) | Always run `.venv\Scripts\python.exe`, never the bare `python`. |
| JDK 20 on PATH (`javac`, `java`) | Compiler feedback uses the real javac. |
| Network is flaky | Nothing in the core path may require a download. Real-corpus pretraining is optional with a local fallback. |
| RL needs thousands of compiles | Spawning `javac` per sample (~0.5-1 s) is too slow. We run one persistent JVM that compiles in memory. |
| Course project on Transformers | The model is implemented in plain PyTorch (`nn.Module`s we write), not a pretrained HuggingFace checkpoint. |

## 3. Repository layout and ownership

```
PLAN.md
pyproject.toml                  (scaffold, do not restructure)
mjava/
  interfaces.py                 SHARED CONTRACT - read first, do not change signatures
  compiler/                     WS-A
    java/CompileServer.java       persistent in-memory javac server
    server.py                     ServerCompiler (JavaCompiler protocol)
    subprocess_compiler.py        SubprocessCompiler fallback
    __init__.py                   get_compiler()
  data/                         WS-A
    spec.py                       spec sampling (classes, fields, methods, ...)
    render_java.py                spec -> canonical Java
    render_nl.py                  spec -> varied natural-language instruction
    mutate.py                     correct Java -> broken Java (for repair data)
    conformance.py                spec_conformance(spec, java) -> float in [0,1]
    build.py                      CLI: writes data/*.jsonl
  tokenizer.py                  WS-B
  model/                        WS-B
    transformer.py                encoder-decoder Transformer
    decoding.py                   greedy / beam / sampling
    generator.py                  TransformerGenerator (CodeGenerator protocol)
  train/                        WS-B (supervised.py, common.py), WS-D (rl.py)
  eval/                         WS-D
  pipeline/                     WS-C
    mjava_file.py                 .mjava parsing / block detection
    correct.py                    self-correction loop
  cli.py                        WS-C
  configs/                      tiny.yaml, cpu.yaml, gpu.yaml  (WS-B)
tests/                          each workstream owns tests/test_<its area>*.py
data/ checkpoints/ runs/        generated, git-ignored
```

Rules for every agent:

- Stay inside the files your workstream owns. If you need something from
  another workstream, code against `mjava/interfaces.py` and use a fake in tests.
- Do not `git commit`, do not create branches, do not `pip install` new
  dependencies without stating why in your final report.
- Tests run with `.venv\Scripts\python.exe -m pytest tests/test_<area>*.py -q`.
  Mark tests that need a JDK with `@pytest.mark.javac`, slow ones with `@pytest.mark.slow`.
- Finish by reporting: what exists, what the tests show (paste the summary
  line), what is not done or is shaky. Report failures as failures.

## 4. Shared contracts (see `mjava/interfaces.py`)

- `JavaCompiler.compile(source) -> CompileResult(ok, diagnostics)`; `compile_many`.
- `CodeGenerator.generate(sources, GenerationConfig) -> list[list[str]]`.
- Encoder source layouts:
  - generate: `<gen> {instruction}`
  - repair: `<fix> {instruction} <err> {error_text} <code> {broken_java}`
- Dataset files, JSON lines:
  - `data/{train,val,test}.jsonl` -> `Example` (`id, instruction, java, spec`)
  - `data/repair_{train,val,test}.jsonl` -> `RepairExample`
  - `data/corpus_{train,val}.txt` -> optional raw Java for LM pretraining, one
    file/snippet per record separated by a line containing only `<|endoftext|>`.

## 5. Workstreams

### WS-A — Compiler service and dataset (wave 1)

**Compiler**

- `CompileServer.java`: single file, no dependencies. Uses
  `javax.tools.ToolProvider.getSystemJavaCompiler()` with in-memory
  `JavaFileObject`s and a file manager that discards class output. Protocol:
  one JSON object per line on stdin `{"id": n, "filename": "...", "source": "..."}`,
  one JSON line on stdout `{"id": n, "ok": bool, "diagnostics": [{kind, line, column, code, message}]}`.
  Hand-roll the JSON (no libraries). Pass `-proc:none -Xlint:none`. A source
  with a `public` class must be given a matching filename by the Python side
  (derive it from the first top-level type name).
- `ServerCompiler`: builds the server class once into
  `mjava/compiler/java/build/` (rebuild if the `.java` is newer), starts
  `java -cp build CompileServer`, speaks the protocol, restarts the JVM if it
  dies or a request exceeds a timeout (default 20 s), is thread-safe via a lock,
  is a context manager. `compile_many` just loops (the JVM stays warm).
- `SubprocessCompiler`: temp dir + `javac`, parses stderr. Fallback only.
- `get_compiler(kind="auto")`: server if it starts, else subprocess.
- Target: >= 100 small compiles/second after warm-up. Measure and report.

**Dataset** — a synthetic, structured generator. Each example is sampled as a
`spec` dict first, then rendered twice (Java and natural language).

- Task families, each with several difficulty knobs:
  1. data class: fields (+ optional constructor, getters, setters, `toString`, `equals`/`hashCode`)
  2. class with simple methods (arithmetic on fields, boolean predicates, counters, accumulators)
  3. static utility methods (sum/max/min/average of array, factorial, fibonacci, isPrime, reverse string, count occurrences, ...)
  4. loops and control flow in a `main` (print 1..n, sum a range, multiplication table, FizzBuzz-style conditions, iterate an array)
  5. interface + implementing class; abstract class + subclass; simple inheritance with `super(...)`
  6. enum with constants (+ optional field and constructor)
  7. multi-instruction examples: 2-3 sentences that together describe one class
- Types: `int, long, double, boolean, char, String`, arrays of these, and
  `List<...>`/`Map<...>` for a minority (with the right imports).
- Names: large pools of class/field/method nouns and verbs so that the test
  split contains name combinations never seen in training. Keep a held-out set
  of identifiers that appear **only** in `test.jsonl` to measure generalisation.
- Natural language: at least 5 distinct phrasings per construct, optional
  field-order and connective variation ("with fields", "having", "that has",
  "name as String" / "a String name" / "name: String"), varying capitalisation
  and trailing punctuation. This is what stops the model from learning one template.
- Canonical Java formatting: 4-space indent, one member per line, deterministic
  member order. Top-level types are package-private so file names never matter.
  Target outputs <= ~350 tokens.
- Every reference Java is compiled with the real compiler during the build;
  anything that fails is a generator bug and must be fixed, not filtered.
- `mutate.py`: produce broken variants with realistic single/double faults —
  drop `;`, drop/duplicate a brace or paren, misspell an identifier at a use
  site, wrong declared type (`int x = "a"`-style mismatch), missing `return`,
  wrong return type, undeclared variable, missing import, `String` -> `string`,
  missing cast. Keep only mutants that actually fail to compile, and record the
  real `error_text`.
- `conformance.py`: `spec_conformance(spec, java) -> float` — fraction of the
  spec's required elements (type name and kind, each field name+type, each
  method name+return type+parameter types, superclass/interfaces, enum
  constants) found in the generated source after comment/whitespace
  normalisation. Regex/token based, no external parser. This is what stops RL
  from rewarding `class A {}`.
- `python -m mjava.data.build --out data --n-train 60000 --n-val 2000 --n-test 2000 --seed 0`
  (`--n-repair-*` too). Deterministic for a given seed. No duplicate
  instructions across splits.

Acceptance: 100% of references compile; mutants 100% fail; `spec_conformance(spec, reference) == 1.0`
for every example and `< 1.0` when a member is removed; build of the default
sizes finishes in a few minutes; tests pass.

### WS-B — Tokenizer, Transformer, supervised training (wave 1)

- `tokenizer.py`: byte-level BPE via the `tokenizers` library, trained on
  instructions + Java, vocab ~6000, the `SPECIAL_TOKENS` from `interfaces.py`
  as atomic tokens. Must round-trip Java exactly (whitespace included).
  `train_tokenizer(paths, out)`, `load_tokenizer(path)`, `encode`, `decode`.
- `model/transformer.py`: encoder-decoder Transformer written with our own
  modules (multi-head attention, pre-LN blocks, learned or sinusoidal
  positions, tied decoder input/output embeddings, shared source/target
  embedding). Do not use `nn.Transformer`. Config dataclass:
  `vocab_size, d_model, n_heads, n_enc_layers, n_dec_layers, d_ff, dropout, max_src_len, max_tgt_len`.
  `forward(src, tgt_in, masks) -> logits`. Must also expose
  `encode(src)` and an incremental `decode_step` with a key/value cache so
  that generation is not quadratic — CPU inference speed matters for RL.
- `model/decoding.py`: batched greedy, beam search, and temperature/top-k
  sampling. Sampling must be able to return per-token log-probs **with
  gradients** (needed by RL) as well as a no-grad fast path.
- `model/generator.py`: `TransformerGenerator.from_checkpoint(dir)` implementing `CodeGenerator`.
- `train/supervised.py`:
  - stage `lm` (optional): decoder-only next-token pretraining on
    `data/corpus_*.txt` if present, else on the `java` field of `train.jsonl`
    (encoder fed an empty `<gen>` source).
  - stage `seq2seq`: generate examples and repair examples mixed (default 70/30),
    cross-entropy with label smoothing 0.1, AdamW, warmup + cosine decay,
    gradient clipping, length-bucketed batches, periodic validation reporting
    **loss and perplexity**, best-checkpoint saving, resumable, a
    `--max-minutes` wall-clock budget, CSV/JSONL metrics log in `runs/<name>/`.
  - `torch.set_num_threads` configurable; default to physical cores.
- `configs/`: `tiny` (2 layers, d=64 — unit tests and smoke runs, must overfit
  64 examples in < 2 min), `cpu` (about d=256, 4+4 layers, 4 heads, ff=1024,
  src<=192, tgt<=384), `gpu` (about d=512, 6+6 layers).
- Checkpoint directory layout: `model.pt`, `config.yaml`, `tokenizer.json`.

Acceptance: unit tests for masking (no future leakage: changing token t+1 must
not change logits at <= t), cached decoding == uncached decoding, tokenizer
round-trip, greedy == beam(1); the `tiny` config overfits a small fixture set to
> 95% exact match. Report measured training throughput (tokens/s) and greedy
generation latency for the `cpu` config so later stages can budget time.
WS-A's data may not exist yet: build tests on a small hand-written fixture in
`tests/fixtures/`.

### WS-C — `.mjava` pipeline and CLI (wave 1)

- `.mjava` format: a file is a sequence of blocks separated by blank lines.
  - Comment lines (`//`, `/* */`) are kept as-is.
  - A block that is already Java (starts with `package`, `import`, an
    annotation, a modifier, `class`/`interface`/`enum`/`record`, or ends with
    `}`/`;` with balanced braces) is passed through unchanged.
  - Anything else is a shorthand block: its lines are joined into one
    instruction. An explicit marker `@gen ...` forces a block to be treated as shorthand.
  - Imports produced by expansions are hoisted and de-duplicated at the top of the output.
- `pipeline/correct.py`: `SelfCorrector(generator, compiler, max_retries=3)`
  - attempt 0: `<gen>` source, greedy/beam.
  - on failure: `<fix>` source with the instruction, `CompileResult.error_text()`
    and the previous attempt.
  - later attempts: repair again from the newest failure, then fall back to
    sampling several candidates and taking the first that compiles; never
    return a candidate identical to one already tried.
  - Each block is first compiled on its own (fast path); the assembled file is
    compiled at the end, and file-level diagnostics are mapped back to the
    block that owns the offending line so only that block is retried.
  - Returns a structured trace: per block, each attempt's code, diagnostics,
    and the strategy used. This trace is also what the evaluation consumes.
- `cli.py` / `mjavac`:
  `mjavac File.mjava [-o File.java] [--checkpoint DIR] [--max-retries N] [--keep-temp] [--verbose] [--run]`
  - works in a temp directory; the input is opened read-only and never rewritten.
  - success: writes the `.java` next to the input (or `-o`), exit 0.
  - failure: exit 1, prints the remaining compiler errors, writes
    `File.mjava.review.txt` with the last attempt and diagnostics ("flag for manual review").
  - `--verbose` prints each attempt and the error fed back.
- Tests use a scripted fake `CodeGenerator` (returns a broken answer first,
  a fixed one when it sees `<fix>`) and a fake compiler, plus `javac`-marked
  integration tests against the real `get_compiler()` when it is importable.

Acceptance: loop terminates within the retry limit in every path; original
file byte-identical after a run; sample `examples/Student.mjava` from the
slides goes end-to-end with the fake generator; tests pass.

### WS-D — RL fine-tuning and evaluation (wave 2, after A and B)

- `train/rl.py`: self-critical sequence training (REINFORCE with a greedy baseline).
  For each batch: greedy decode `y_g` (no grad), sample `y_s` (with log-probs),
  compile both, `loss_rl = -(R(y_s) - R(y_g)) * sum(log p(y_s))`, normalised by length.
  - Reward: `R = conformance` if it compiles; otherwise
    `0.3 * conformance * max(0, 1 - 0.2 * n_errors)`. Truncated output (no `<eos>`) gets 0.
    Pure "does it compile" reward is degenerate (an empty class always compiles),
    hence the conformance factor.
  - Total loss `= lambda_rl * loss_rl + (1 - lambda_rl) * loss_mle` on the same
    batch (default `lambda_rl = 0.5`) to stop the policy drifting away from Java.
  - Applies to both generate and repair sources.
  - Logs mean sampled reward, greedy compile rate, greedy conformance per step;
    validates on a fixed subset; keeps the best checkpoint by greedy
    compile-and-conform rate; `--max-minutes` budget; low learning rate (~1e-5).
- `eval/evaluate.py`: on `test.jsonl` report
  - perplexity (teacher-forced),
  - first-attempt compile rate, exact match, mean conformance,
  - with the self-correction loop: success rate after 1, 2, 3 retries and mean attempts,
  - the same split by task family and for the held-out-identifier subset,
  - a retrieval/template baseline for comparison (nearest training instruction's Java with names substituted),
  - writes `runs/<name>/eval.json` and a markdown table; a plotting script
    produces training-loss, perplexity, and RL-reward curves as PNGs.

Acceptance: RL step verified on the `tiny` model with a fake reward (gradient
sign test: a rewarded sample's probability goes up); a short real run does not
reduce the greedy compile rate on validation.

### WS-E — Training runbook and integration check (wave 3)

Training is run by the user on the GPU machine, not by an agent. This
workstream delivers everything needed to do that in one go:

1. `scripts/run_all.py` (cross-platform, resumable per stage): environment
   check (CUDA, JDK) -> dataset build -> tokenizer -> supervised training (`gpu`
   config) -> RL fine-tuning -> evaluation of supervised-only vs. supervised+RL,
   with and without the correction loop -> plots. Each stage is also runnable
   on its own.
2. An end-to-end dry run on this machine with the `tiny` config and a few
   hundred examples (seconds per stage) to prove the stages chain together and
   that a produced checkpoint drives `mjavac examples/Student.mjava`. Output
   quality is irrelevant here; only that nothing crashes.
3. `README.md`: setup on the GPU machine (CUDA torch wheel, JDK), the exact
   commands, expected durations as estimates, how to copy `checkpoints/` back
   and run `mjavac` on CPU, known limitations. The results table is left to be
   filled from `runs/<name>/eval.json` after the real run.

## 6. Order of execution

```
wave 0 (done)   scaffold, interfaces.py, venv
wave 1          WS-A   WS-B   WS-C        in parallel, disjoint files
wave 2          WS-D   RL and evaluation code (tested on `tiny` only)
wave 3          WS-E   run_all script, tiny dry run, README
then            user runs the real training on the GPU machine
```

## 7. Risks

- **GPU code paths are untested until the user's first run** — keep them
  simple (standard autocast/GradScaler), make every stage resumable, and have
  `run_all.py` start with a 1-minute `tiny` sanity pass on the GPU so a crash
  shows up immediately instead of hours in.
- **RL collapses output quality** — MLE mixing, small LR, best-checkpoint
  selection on validation, conformance in the reward.
- **Model only memorises templates** — phrasing variety, held-out identifier
  split, per-family reporting make this visible instead of hidden.
- **Compile server hangs** — per-request timeout and automatic restart;
  subprocess fallback.
