# mjava: natural-language shorthand to compilable Java

> **Setting this up on a new machine (especially Windows)? Start with [`WINDOWS_LOCAL_SETUP.md`](WINDOWS_LOCAL_SETUP.md).**

A Transformer written from scratch in PyTorch (no pretrained model, no `nn.Transformer`) expands natural-language
shorthand into Java. You write a `.mjava` file such as

```
// Custom syntax a user would actually write --
Create a class Student with fields: name as String,
  age as int, gpa as double.
```

and the `mjavac` command turns it into a `.java` file. It compiles the model's output with the real `javac`; if that
fails, the compiler errors are fed back to the model, which produces a corrected version, up to a retry limit. The model is
trained supervised first (generate examples plus repair examples "broken code + compiler error -> fixed code") and then
fine-tuned with reinforcement learning (self-critical REINFORCE) using a compiler-derived reward.

Authors: Aayush Shukla (23BPS1081), Karan Kumar (23BPS1034), Aryan Kumar (23BPS1193).

**Status in one line: a small model (the `cpu` config, 4.4M parameters) has been fully trained and evaluated on an Apple M4 CPU
(see "Results"); no stage has ever run on a GPU, and the larger `gpu` / `gpu4` configs have not been trained yet. See "Status and
known limitations" at the end.**

## Repository map

```
scripts/run_all.py        one command that runs every stage in order (resumable)
mjava/
  interfaces.py           shared contracts (compiler / generator protocols, model input format)
  compiler/               persistent in-memory javac server (CompileServer.java) + Python client, subprocess fallback
  data/                   synthetic dataset generator: spec -> Java and natural language; mutations for repair data;
                          spec_conformance(); build.py is the CLI
  tokenizer.py            byte-level BPE (HuggingFace `tokenizers`)
  model/                  encoder-decoder Transformer, greedy/beam/sampling decoding, TransformerGenerator
  train/                  common.py (data, batching, config), supervised.py (stages lm, seq2seq), rl.py
  pipeline/               .mjava parsing and the self-correction loop
  cli.py                  the `mjavac` command
  eval/                   evaluate.py, baseline.py (retrieval baseline), compare.py (tables), plots.py (PNG curves)
  configs/                tiny.yaml (tests/smoke), cpu.yaml (untuned fallback), gpu.yaml (the real one, 8-12 GiB), gpu4.yaml (4 GiB cards such as an RTX 3050; untested on CUDA); each has a `rl:` section
examples/                 Student.mjava, Geometry.mjava, Library.mjava
tests/                    pytest suite
data/                     generated dataset (git-ignored, already built on the dev machine)
checkpoints/ runs/        training outputs (git-ignored)
PLAN.md                   design document
```

## Setup on the GPU machine

You need Python 3.11 or newer, a JDK 11 or newer on `PATH` (`javac -version` and `java -version` must both work), and an
NVIDIA GPU with a working driver (`nvidia-smi` shows it). Windows and Linux both work; nothing in the code is OS specific.

Windows (PowerShell), from the repo root:

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
# 1) install the CUDA build of torch, see below
# 2) then:
pip install -e .[dev]
```

Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
# 1) install the CUDA build of torch, see below
pip install -e '.[dev]'
```

**Install torch first, from the PyTorch index that matches your CUDA driver.** Do not copy a `cuXXX` tag from anywhere
(including this file): open <https://pytorch.org/get-started/locally/>, pick your OS, Pip, Python and the CUDA version the
page offers for your driver, and run the command it prints (of the form
`pip install torch --index-url https://download.pytorch.org/whl/<tag-chosen-on-that-page>`). Installing plain
`pip install torch` on Windows gives a CPU-only wheel. `pip install -e .[dev]` afterwards leaves that torch in place and adds
`tokenizers`, `numpy`, `pyyaml`, `tqdm`, `pytest`, `matplotlib` (used for the plots) and `datasets`.

Check it:

```
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

must print `True` and the GPU name. The dataset: copy the `data/` directory from the dev machine (see the limitation about
JDK versions below); if there is no `data/meta.json` the first stage of `run_all.py` builds it (several minutes).

## Run everything

From the repo root, with the venv active:

```
python scripts/run_all.py
```

On the first line it prints what it found (Python, torch/CUDA, GPU name and VRAM, JDK, free disk) and stops with a clear
message if something is missing. If CUDA is available it then runs the smoke preset (below) first, so a broken CUDA, AMP or
JDK path fails in the first minute instead of hours in (`--skip-smoke` skips that). Then, in order:

| stage | what it does | default time budget |
|---|---|---|
| `env` | environment check, see above | seconds |
| `data` | `mjava.data.build` (skipped if `data/meta.json` exists) | minutes, only if `data/` is missing |
| `tokenizer` | BPE tokenizer, vocab 6000, to `checkpoints/tokenizer.json` | seconds |
| `lm` | optional decoder-only pretraining, only with `--lm` | `--lm-minutes` (30) |
| `seq2seq` | supervised training on generate + repair examples | `--s2s-minutes` (180) |
| `rl` | RL fine-tuning from the seq2seq checkpoint | `--rl-minutes` (90) |
| `eval_retrieval`, `eval_s2s`, `eval_rl` | test-set evaluation of the retrieval baseline, the supervised model, the RL model (with the self-correction loop) | not measured |
| `compare` | one markdown table from the three `eval.json` | seconds |
| `plots` | training curves as PNG | seconds |
| `best` | picks the better of supervised and supervised+RL from the evaluation and copies it to `checkpoints/best` | seconds |
| `demo` | runs `mjavac` on `examples/Student.mjava` using `checkpoints/best` with no flags | seconds |

The budgets are wall-clock caps (the learning-rate schedule follows the clock, so a run that hits the cap still ends with a
decayed schedule); they are defaults for a multi-hour run, not measured durations. We have **not** measured how long any stage
takes on a GPU. Training also stops earlier at the config's `train.max_steps` (20000) / `rl.max_steps` (2000). Change the
budgets with `--s2s-minutes 120 --rl-minutes 45` etc., and look at the first minutes of the log (steps per second) to
extrapolate.

The `best` stage compares the two models on the test set by the fraction of examples that compile **and** satisfy the spec
after the self-correction loop (ties go to first-attempt rate, and a tie keeps the supervised model). It prints both numbers
and states whether RL helped or not; it does not assume that it did. If the RL checkpoint is byte-identical to the supervised one
(RL never beat the supervised model on its validation subset) it says so and keeps the supervised one.

Useful flags: `--config gpu|gpu4|cpu|tiny` (default gpu), `--device auto|cpu|cuda`, `--no-amp`, `--eval-limit N` (evaluate only the
first N test examples, handy for a first look), `--lm`, `--set key=value` (repeatable, forwarded to the trainers, e.g.
`--set train.batch_tokens=8000 --set rl.lr=2e-5`), `--data DIR`, `--allow-cpu`. `python scripts/run_all.py --help` lists all of them.

### Single stages, skip and resume

```
python scripts/run_all.py --only seq2seq          # also: a comma list, or the aliases train / eval
python scripts/run_all.py --from rl               # rl and everything after it
python scripts/run_all.py --skip lm,plots
python scripts/run_all.py --force --only rl       # redo a finished stage (training restarts from scratch)
```

Progress is recorded in `runs/run_all_state.json`. A finished stage is skipped on re-run (it also checks that its output still
exists). If a training stage was interrupted (Ctrl-C, closed terminal, crash, reboot), run the same command again: the script
passes the trainer's own `--resume`, which continues from `checkpoints/<name>/last.pt` (saved at every validation), and the time
budget counts all sessions together. When a stage is re-run, the stages that depend on it are marked stale and run again. If a
stage fails, the script prints the exact command, which you can run by hand, and stops.

### Smoke preset (also usable on any machine)

```
python scripts/run_all.py --preset smoke
```

Tiny model, a throw-away dataset of 300/60/60 examples, budgets of seconds per training stage. It writes only to
`runs/smoke/` and `checkpoints/smoke/` (never to `data/`, `checkpoints/gpu_*` or `runs/gpu_*`) and runs every stage including
`lm`, evaluation, compare, plots, `best` and the `mjavac` demo. The model it produces is garbage; the point is that the stages
chain together. On the dev machine (CPU, no GPU) it took 2.4 to 3.6 minutes (two runs), of which the three evaluations are
the largest part (run on 40 test examples); a re-run skips every finished stage.

## The stages by hand

Everything `run_all.py` does is a plain command (shown for the `gpu` config; `run_all.py` prints the exact commands it uses):

```
python -m mjava.data.build --out data --n-train 300000 --n-val 3000 --n-test 3000 --n-repair-train 90000 --n-repair-val 1500 --n-repair-test 1500 --seed 0
python -m mjava.tokenizer --data data --out checkpoints/tokenizer.json --vocab-size 6000

# optional pretraining, then supervised training (--init-from continues from the lm model)
python -m mjava.train.supervised --stage lm --config gpu --name gpu_lm --max-minutes 30
python -m mjava.train.supervised --stage seq2seq --config gpu --name gpu_s2s --max-minutes 180 [--init-from checkpoints/gpu_lm]
python -m mjava.train.supervised --stage seq2seq --config gpu --name gpu_s2s --resume --max-minutes 180   # continue

python -m mjava.train.rl --config gpu --init-from checkpoints/gpu_s2s --name gpu_rl --max-minutes 90
python -m mjava.train.rl --config gpu --init-from checkpoints/gpu_s2s --name gpu_rl --resume --max-minutes 90

python -m mjava.eval.evaluate --baseline retrieval --name retrieval
python -m mjava.eval.evaluate --checkpoint checkpoints/gpu_s2s --name gpu_s2s
python -m mjava.eval.evaluate --checkpoint checkpoints/gpu_rl  --name gpu_rl
python -m mjava.eval.compare runs/retrieval runs/gpu_s2s runs/gpu_rl --labels "retrieval baseline,supervised,supervised+RL" --out runs/compare.md
python -m mjava.eval.plots --run runs/gpu_s2s --run runs/gpu_rl --out runs/plots
```

Trainer options: `--set k=v` (config override, e.g. `train.lr=1e-3`; `1e-3` style numbers are read as floats), `--device`,
`--no-amp` (disable mixed precision), `--max-steps`, `--threads`, `--ckpt-root`, `--runs-root`. Evaluation options: `--split`,
`--limit N`, `--max-retries`, `--batch-size`.

## What is written where

| path | content |
|---|---|
| `data/{train,val,test}.jsonl`, `data/repair_{train,val,test}.jsonl`, `data/corpus_*.txt`, `data/meta.json` | dataset (300k/3k/3k generate, 90k/1.5k/1.5k repair; about 6.5 minutes to build) |
| `checkpoints/tokenizer.json` | the tokenizer |
| `checkpoints/gpu_lm/`, `gpu_s2s/`, `gpu_rl/` | `model.pt`, `config.yaml`, `tokenizer.json` = best checkpoint by validation (loads on CPU); `last.pt` = resume state with optimizer |
| `checkpoints/best/` | the chosen final model (`model.pt`, `config.yaml`, `tokenizer.json`, `SOURCE.txt` saying where it came from) |
| `runs/<name>/metrics.jsonl`, `metrics.csv` | training and validation log per run |
| `runs/<name>/eval.json`, `eval.md`, `eval_examples.jsonl` | evaluation (the training run directory is reused for its evaluation); `runs/retrieval/` for the baseline |
| `runs/compare.md` | the results table (overall, by task family, held-out identifiers) |
| `runs/plots/*.png` | loss, perplexity, RL reward, greedy compile rate, RL validation curves |
| `runs/run_all_state.json` | which stages are done |
| `runs/demo/Student.java` | output of the final `mjavac` demonstration |

## Is it going well? What to look at

Supervised training (`[val]` lines and `step` lines in the log, `runs/gpu_s2s/metrics.csv`):

- `loss` on the training lines should fall quickly in the first few hundred steps; `[val] ... ppl` (validation perplexity,
  without label smoothing) should keep falling and then flatten; we have no real-scale reference value yet (the 20-second tiny
  smoke model gets validation perplexity of about 9 on its 300 examples, which means nothing for the real run).
  `(best, saved)` marks a new best checkpoint.
- If `loss` becomes `nan`, or val ppl rises while train loss falls, lower `train.lr` (`--set train.lr=2.5e-4`).
- `tgt-tok/s` and `step/s` are the throughput; use them to estimate how far the time budget gets you.

RL (`step` lines and `[val]` lines in the RL log, `runs/gpu_rl/metrics.csv`):

- The first line `[val baseline] step 0 ...` is the supervised model: its greedy `compile` rate, `conf` (spec conformance) and
  `full` (compiles and fully conformant). RL only keeps a checkpoint that beats this on `full`, then `compile`, then `conf`.
- `adv!=0` (`adv_nonzero_frac`) is the share of sampled outputs whose reward differs from the greedy output's. If it is
  stuck near 0, RL has nothing to learn from (the model is already deterministic or always right/wrong): raise `rl.temperature` or
  `rl.num_samples`. Near 1 with very noisy rewards is also not useful.
- `R` (mean sampled reward) vs `Rg` (greedy reward): the sampled one is expected to be a little lower; the trend of both should be up.
- `greedy-compile` and `conf` on the training batches, and the `[val]` greedy `compile`/`full` against the baseline: the real
  signal. `ppl` in the RL `[val]` lines should not blow up (the MLE term is there to hold it).
- `no-spec` should be 0.00. If it is large the dataset is not the one from `mjava.data.build`.

If RL does not improve the validation numbers, that is a legitimate outcome; the `best` stage then uses the supervised model.

## Dataset v2: generated identifiers

About 55% of the class / field / method / interface names in the data are generated (real-word compounds and invented words,
`mjava/data/synth.py`) instead of coming from the small fixed pools, so that the model has to copy names from the instruction
instead of memorising them. A run trained on the first dataset (60k examples, fixed pools) compiled and matched the spec for 91%
of examples with names it had seen but only 1.3% with novel names. Held-out test names are chosen by hashing the whole identifier
(`synth.is_held`), so training never contains them. Because generated names appear in both halves of the test set, the
"seen identifiers" row is no longer purely seen names; the held-out row stays strictly novel. 20% of interface examples are bare
interfaces without an implementing class.

## CUDA out of memory

`gpu.yaml` is set for a single 8-12 GiB card (bf16 if the card supports it, otherwise fp16 with a gradient scaler). If a stage
dies with `CUDA out of memory`, `run_all.py` prints the knobs; re-run the same command (finished stages are skipped, an
interrupted one resumes):

| stage | knob | default | try |
|---|---|---|---|
| supervised | `train.batch_tokens` (padded target tokens per batch) | 16000 | 10000, then 8000, 4000 |
| supervised | `train.max_batch_examples` | 128 | 64 |
| RL | `rl.batch_examples` (sources per batch) | 32 | 16, 8 |
| RL | `rl.num_samples` (samples per source) | 2 | 1 |
| RL | `rl.batch_tokens` | 16000 | 8000 |

e.g. `python scripts/run_all.py --set train.batch_tokens=8000 --set rl.batch_examples=16 --set rl.num_samples=1`.
With 24 GiB or more you can raise `train.batch_tokens` (try 32000) for speed. Do not lower `model.max_src_len` (448) or
`max_tgt_len`: repair sources reach about 380 tokens and targets about 280. The RL trainer skips a batch that runs out of memory
(and logs it) up to `rl.max_oom_skips` times before aborting. Changing batch settings changes the optimisation slightly; if the
OOM happens at the very start, nothing is lost. `--no-amp` turns off mixed precision if you suspect it of causing NaNs.

## Using the trained model on a CPU-only laptop

Copy `checkpoints/best/` (three small files, no GPU needed to load) and the `examples/` folder to the laptop, set up the repo
there (`pip install -e .` with a normal CPU torch, JDK 11+), keep the same folder layout so that the default
`checkpoints/best` path works, and from the repo root:

```
mjavac examples/Student.mjava            # writes examples/Student.java next to the input, exit code 0
mjavac examples/Geometry.mjava -o out/Geometry.java --verbose   # --verbose shows each attempt and the error fed back
mjavac examples/Library.mjava --max-retries 5 --run              # --run also runs main() of the result
mjavac some.mjava --checkpoint D:\models\best                    # another model directory
```

If `mjavac` is not on `PATH`, use `python -m mjava.cli ...` instead. The default checkpoint is `$MJAVA_CHECKPOINT` if set,
else `checkpoints/best` relative to the current directory. Exit codes: 0 compiled and written; 1 still not compiling after the
retries (the input is untouched and `<file>.mjava.review.txt` holds the last attempt and the compiler errors, "flagged for
manual review"); 2 usage or environment error (missing checkpoint, no JDK, crash); 3 only with `--run`: the program compiled but
failed when run. The `.mjava` input is never modified.

## The `.mjava` format in brief

A file is a sequence of blocks separated by blank lines.

- Comment lines (`//`, `/* */`) are kept as they are.
- A block that already is Java (starts with `package`, `import`, an annotation, a modifier or `class`/`interface`/`enum`/`record`
  and has a body, or ends with `}`/`;` with balanced braces and Java punctuation) is passed through unchanged.
- Anything else is shorthand: its lines are joined into one instruction for the model (a block can span several lines).
  Start a block with `@gen` to force it to be treated as shorthand.
- Imports in the output are collected at the top and de-duplicated; a `package` line is kept first.
- The model was trained on instructions like "Create a class X with fields: a as int, b as String.", "Write a static method
  isEven(int n) that returns true when n is even.", "an enum Color with constants RED, GREEN and BLUE", interfaces,
  abstract classes and inheritance, and main programs with loops. See `examples/` and `mjava/data/render_nl.py` for the phrasings.

## Results

All numbers below come from one complete `scripts/run_all.py --config cpu` run on an Apple M4 (CPU only, 24 GB), trained on the
current dataset (300k generate + 90k repair training examples, generated identifiers, see "Dataset v2"). Settings: 180 min
supervised training (20,704 steps at about 2.2 steps/s and 7-8k target tokens/s, a little over one pass through the 390k
training examples; validation perplexity 1.07 on the training distribution), about 3 min RL (stopped by `rl.max_steps=300`), evaluation on the **first 600 of the 3000 test examples** with up to 3 correction retries.
Total wall-clock 3.1 h (supervised 180 min, RL 3.4 min including loading, three evaluations about 2 min). The model is the `cpu` config: 4.4M parameters, d_model 192, 3+3 layers. These are the written-out
`runs/compare.md` tables; `checkpoints/best` is the supervised+RL checkpoint (the better of the two by the final compile+conform
rate, by a margin that is within noise for n=600).

### Overall

| system | n | ppl | 1st compile | exact | conformance | compile+conform | success@0 | success@1 | success@2 | success@3 | mean attempts | final conformance | final compile+conform | repair compile |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| retrieval baseline | 600 | - | 85.8% | 4.2% | 28.5% | 7.3% | 85.8% | 85.8% | 85.8% | 85.8% | 1.00 | 28.5% | 7.3% | - |
| supervised | 600 | 1.174 | 83.7% | 69.8% | 93.2% | 71.8% | 83.7% | 86.5% | 87.3% | 87.5% | 1.42 | 93.1% | 73.0% | 82.8% |
| supervised+RL | 600 | 1.147 | 83.7% | 70.0% | 93.0% | 72.0% | 83.7% | 85.5% | 86.3% | 87.2% | 1.44 | 93.1% | 73.2% | 82.0% |

### By task family (compile-and-conformant rate: first attempt / after the correction loop)

| family (n) | retrieval baseline: 1st / final compile+conform | supervised: 1st / final compile+conform | supervised+RL: 1st / final compile+conform |
|---|---:|---:|---:|
| abstract_class (37) | 5.4% / 5.4% | 64.9% / 64.9% | 64.9% / 64.9% |
| class_methods (103) | 0.0% / 0.0% | 59.2% / 60.2% | 57.3% / 60.2% |
| data_class (95) | 0.0% / 0.0% | 72.6% / 73.7% | 72.6% / 72.6% |
| enum (51) | 21.6% / 21.6% | 68.6% / 70.6% | 70.6% / 70.6% |
| inheritance (55) | 0.0% / 0.0% | 60.0% / 63.6% | 60.0% / 63.6% |
| interface_impl (39) | 0.0% / 0.0% | 71.8% / 71.8% | 74.4% / 74.4% |
| main_program (55) | 49.1% / 49.1% | 92.7% / 94.5% | 92.7% / 94.5% |
| multi_instruction (92) | 0.0% / 0.0% | 70.7% / 70.7% | 71.7% / 72.8% |
| static_utils (73) | 5.5% / 5.5% | 89.0% / 90.4% | 89.0% / 89.0% |

### Held-out identifier subset

| subset (n) | retrieval baseline: 1st compile / 1st conform / final compile+conform | supervised: 1st compile / 1st conform / final compile+conform | supervised+RL: 1st compile / 1st conform / final compile+conform |
|---|---:|---:|---:|
| seen identifiers (287) | 87.1% / 30.8% / 9.8% | 97.9% / 98.7% / 92.3% | 98.3% / 98.8% / 93.0% |
| held-out identifiers (313) | 84.7% / 26.4% / 5.1% | 70.6% / 88.1% / 55.3% | 70.3% / 87.7% / 55.0% |

How to read this:

- **The self-correction loop helps:** compiling code on the first attempt is 83.7%, after up to 3 retries 87.5%.
- **The model copes with names it has never seen, but not reliably:** for held-out identifiers (never in training) 55.0% of
  examples end up compiling *and* matching the spec, against 93.0% for names that occur in training. The typical failure is a
  spelling slip when a long invented name is repeated (for example `quillTrance` becoming `quillTancerance` inside the
  constructor body).
- **RL changed almost nothing** (72.0% vs 71.8% first attempt, 73.2% vs 73.0% final; well inside the noise of 600 examples).
- **The retrieval baseline** (nearest training instruction) usually compiles but almost never matches the spec: it copies code for
  a different request.
- Evaluating only 600 examples gives roughly +-3.5 percentage points of sampling error on the overall rates, more on the families.

**Comparison with the first model** (same architecture, trained on the older fixed-name dataset of 60k examples and evaluated on
the *current* test set, first 600 examples): compile+conform 5.8% overall, 12.2% with seen names, 0.0% with held-out names,
against 73.2% / 93.0% / 55.0% above. On its own, older test set the first model had reached 44% overall but only 1.3% with novel
names. The difference is almost entirely the data: with the fixed identifier pools of the first dataset the model memorised names
instead of copying them.

Column meanings: `ppl` = teacher-forced test perplexity; `1st compile` = first attempt compiles; `exact` = equal to the reference
modulo whitespace; `conformance` = mean `spec_conformance` (does the code contain the class, fields, methods and signatures the
instruction asked for); `compile+conform` = compiles and conformance 1.0; `success@k` = compiles after at most k retries of the
self-correction loop; `final ...` = the same measured on the code the loop ends with; `repair compile` = greedy repair of the
repair test set compiles. The training curves are in `runs/plots/` (`loss.png`, `perplexity.png`, `rl_*.png`) after a run.

## Status and known limitations

- **No stage has run on a GPU, and only the small `cpu` config has been trained.** The CUDA code paths (device placement, pinned memory, bf16/fp16 autocast, GradScaler, OOM
  handling) are written with the standard PyTorch idioms but are untested; that is why `run_all.py` starts with a short smoke run
  on the GPU. Nothing has been trained at the `gpu` / `gpu4` scale, so there are no results, durations or throughput figures for them.
- What has been verified, on CPU only: the 145+ unit tests, the `tiny` config end to end through `scripts/run_all.py --preset smoke`
  (data build, tokenizer, lm, seq2seq, RL, three evaluations, compare, plots, best-checkpoint selection, `mjavac` demo, skip and
  resume logic). A tiny model produces garbage, so this shows the plumbing, not quality.
- **The dataset is synthetic.** It is generated from templates (data classes, classes with methods, static utilities, loops in
  `main`, interfaces/abstract classes/inheritance, enums, multi-sentence descriptions) with varied natural-language phrasing and
  large identifier pools; a part of the test set uses identifiers never seen in training. Results therefore measure
  generalisation inside this distribution, not the ability to handle arbitrary Java requests.
- **Regenerating the data on a machine with a different JDK can change the compiler error wording** in the repair examples
  (the `error_text` comes from the real `javac`), and the model is trained on that wording. Either copy `data/` from the dev machine
  (preferred) or let `run_all.py` regenerate it, and then train and evaluate consistently on that one dataset. Do not mix a
  checkpoint trained on one data version with an evaluation or `mjavac` run against a very different JDK without expecting some loss.
- **RL has only been verified mechanically on a toy model** (gradient-sign test on `tiny`, a few-step run in the smoke preset).
  Whether it improves a real model is unknown; the pipeline measures that and the `best` stage does not assume it.
- Only the first CUDA device is used. Evaluation of 2000 test examples with up to 3 correction retries each has not been timed;
  use `--eval-limit` for a quick look.
- The cpu config is a fallback that has not been tuned; the `rl:` sections of the configs list the RL settings (unknown keys raise).
- The timing assertion `tests/test_compiler.py::test_throughput_floor` is skipped by default because it fails on a busy
  machine; set `MJAVA_PERF=1` to run it (the compile server does roughly 90-130 compiles per second on an idle dev machine).
- `spec["base_family"]` stored on `multi_instruction` examples in the generated data is not used by any code and may be
  unreliable; it was left alone so that the generated dataset does not change.

## Tests

```
python -m pytest -q          # a few minutes; needs the JDK
```
