# START HERE: instructions for the AI agent (and the human) running the training

You are an AI coding agent working on a laptop with an NVIDIA GPU. Your job is to **train the `mjava` model on this
machine and return the results**. Everything needed is in this folder. Read this whole file before running anything.

## 0. What this project is

`mjava` is a small encoder-decoder Transformer, written from scratch in PyTorch, that turns natural-language shorthand into
compilable Java ("Create a class Student with fields: name as String, age as int."). `scripts/run_all.py` runs the whole
pipeline in order: environment check, tokenizer, supervised training, reinforcement-learning fine-tuning, evaluation with the
real `javac`, results table, plots, best-checkpoint selection. It is resumable. `README.md` has the full documentation; this
file is the short path to a finished run.

The dataset in `data/` (300k examples, deliberately full of novel identifiers so the model must learn to copy names) is
already built. The goal of this run is a model that still works on names it has never seen.

## 1. Rules

1. **Do NOT rebuild, regenerate, edit or delete `data/`.** The compiler-error wording in the repair examples depends on the
   JDK that built it (JDK 17). Re-generating it here would change the training data.
2. **Do NOT edit files under `mjava/`, `scripts/` or `mjava/configs/`** to "improve" things. Change behaviour only with the
   command-line flags listed below. The one exception: if a CUDA-specific bug blocks the run (the CUDA code paths have never
   been executed on a GPU before), you may make the smallest possible fix. Then write every edited file and the reason into a
   file `CHANGES.md` in the repo root, and include `CHANGES.md` and the edited files in the results.
3. **Do not upload anything anywhere** (no git push, no cloud storage, no web services). The only output is the local
   `results.zip` described in section 7.
4. **Do not kill or restart the training once it is running** unless it has crashed or hung (section 6). Interruptions are
   safe, since re-running the same command resumes, but they waste time.
5. **Report honestly.** If a stage fails or the results are poor, say so with the real output. Never invent numbers.
6. Training takes hours. Run it **in the background with output going to a log file** and check on it at sparse intervals
   (every 10-20 minutes is plenty). Do not block your shell on it, and do not print the whole log each time.

## 2. Ask the human first (only these two things)

- Is the laptop **plugged into power** and set to **never sleep while plugged in** (Windows: Settings > System > Power > Screen
  and sleep > "When plugged in, put my device to sleep after: Never"; also close nothing important, but do not close the lid
  unless "lid close action" is set to "do nothing")? If the laptop sleeps, training pauses; it resumes afterwards, but slowly.
- How many **hours** are available? The default budget below is about 5-6 hours in total. If less is available, lower
  `--s2s-minutes` (e.g. 120 for about 3 hours total).

## 3. Check the machine

Run these and read the output. Each must succeed before you continue.

```
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python --version        # on Windows also try: py -3 --version      (need 3.11 or newer)
javac -version
java -version
```

- No `nvidia-smi` / no NVIDIA GPU: stop and tell the human; this run needs an NVIDIA GPU.
- Python older than 3.11: install a newer Python (3.11-3.13) and use it for everything below.
- No JDK (`javac` or `java` missing): install JDK 17 (Temurin 17 from https://adoptium.net, or `sudo apt install
  openjdk-17-jdk`, or `winget install EclipseAdoptium.Temurin.17.JDK`), open a new terminal and re-check. JDK 11 or newer
  works, 17 matches the dataset.
- Free disk: at least 10 GB.

Write down the GPU name and `memory.total` (in MiB). It decides the config in section 5.

## 4. Set up Python

From the repository root (the folder containing this file). To avoid shell-activation problems, **call the venv's Python by
path** instead of activating the environment.

Linux / macOS:
```
python3 -m venv .venv
PY=.venv/bin/python
```
Windows (PowerShell):
```
py -3 -m venv .venv
$PY = ".venv\Scripts\python.exe"
```

Install **PyTorch with CUDA first**. Do not guess the CUDA tag: open https://pytorch.org/get-started/locally/, choose
Stable, your OS, Pip, Python, and a CUDA version that the driver version from `nvidia-smi` supports, and run the exact
command it prints with `$PY -m` in front, i.e. `$PY -m pip install torch --index-url https://download.pytorch.org/whl/<tag>`.
(A plain `pip install torch` on Windows installs a CPU-only build, which will not work.) Then:

```
$PY -m pip install -e ".[dev]"
$PY -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The last command **must print `True` and the GPU name**. If it prints `False`: the torch build is CPU-only or does not match
the driver. Reinstall torch with the correct index URL (`$PY -m pip uninstall -y torch` first) before doing anything else.

## 5. Choose the config and start training

Pick by the GPU memory you noted (1 GiB = 1024 MiB):

| GPU memory | config | notes |
|---|---|---|
| under 7 GiB (e.g. RTX 3050 4 GB, 3050/3060 laptop 6 GB) | `gpu4` | ~27M-parameter model, small batches |
| 7 to 19 GiB (e.g. 3060 12 GB, 4060, 4070, 4080 laptop 12 GB) | `gpu` | ~44M-parameter model |
| 20 GiB or more | `gpu` | add `--set train.batch_tokens=32000` |

Default command (replace `<CONFIG>`; keep the other flags). Linux:
```
nohup $PY scripts/run_all.py --config <CONFIG> --s2s-minutes 240 --rl-minutes 45 --eval-limit 500 > train_gpu.log 2>&1 &
```
Windows (PowerShell):
```
Start-Process -FilePath $PY -ArgumentList "scripts/run_all.py","--config","<CONFIG>","--s2s-minutes","240","--rl-minutes","45","--eval-limit","500" -RedirectStandardOutput train_gpu.log -RedirectStandardError train_gpu.err.log -WindowStyle Hidden
```

What happens: it prints an environment summary, runs a short **smoke test** first (about a minute; it exists to expose a
broken CUDA/AMP/JDK setup early), then builds the tokenizer, trains (`seq2seq`), runs RL fine-tuning, evaluates three
systems on the test set, and writes the results. The data stage is skipped because `data/meta.json` exists. The time budgets
are wall-clock caps and the learning-rate schedule follows the clock, so a run that reaches its cap still ends cleanly.

## 6. Monitor, and what to do when something goes wrong

Check progress sparsely (Linux `tail -n 5 train_gpu.log`; Windows `Get-Content train_gpu.log -Tail 5`). To see validation
lines only: Linux `grep "\[val\]" train_gpu.log | tail -3`, Windows `Select-String "\[val\]" train_gpu.log | Select-Object -Last 3`.
Stderr (Windows: `train_gpu.err.log`) holds tracebacks.

**Healthy:** after the smoke test, `step N ... loss X ... step/s` lines appear; `loss` falls from about 12 to about 2 within a
few hundred steps; `[val] ... ppl` falls and `(best, saved)` appears. Note the `step/s` value in the report.

**Smoke test fails** (before real training starts): read the message. Typical causes: CPU-only torch (section 4), missing JDK
(section 3), driver too old for the CUDA build (install a torch build for an older CUDA tag).

**`CUDA out of memory`:** re-run the same command with smaller batches added, trying the next row only if the previous one
still fails. Finished stages are skipped and training resumes.

| try | extra flags |
|---|---|
| 1 | `--set train.batch_tokens=4000 --set train.max_batch_examples=32 --set rl.batch_examples=4 --set rl.num_samples=1` |
| 2 | `--set train.batch_tokens=3000 --set train.max_batch_examples=24 --set rl.batch_examples=4 --set rl.num_samples=1 --set rl.batch_tokens=3000` |

Do not lower `model.max_src_len` or `model.max_tgt_len` (the repair examples need them).

**`loss` is `nan`, or validation perplexity rises while training loss falls:** re-run with `--set train.lr=2.5e-4`.

**The process died or the log has not changed for 15+ minutes:** check that the process is gone (Linux `pgrep -f run_all.py`;
Windows `Get-Process python`) and that the GPU is idle (`nvidia-smi`). Then **run the exact same command again**: it resumes from
the last checkpoint. If the same crash repeats twice, stop and apply the "other crash" path below.

**Any other crash or traceback:** copy the last 40 lines of the log and the contents of `runs/run_all_state.json`. Apply the
rules in section 1 (smallest fix only, documented in `CHANGES.md`), or stop and report to the human with that text.

**Done** looks like: the log ends with a `summary` table in which every stage is `ok` (the `lm` stage says `not enabled`, which
is normal and `data` says `skipped (exists)`), followed by `Best checkpoint: ...` and `Results table: ...`.

## 7. Collect the results

When the run is finished, pack the results with the helper script (cross-platform, run from the repo root):

```
$PY scripts/pack_results.py          # add --extra <edited files> if you changed any code
```

It writes `results.zip` containing `checkpoints/best/` (the trained model), `runs/compare.md` (the results table),
`runs/plots/`, all `runs/*/metrics.csv` and `eval.md`, `runs/run_all_state.json`, `train_gpu.log` (and `train_gpu.err.log`),
and `CHANGES.md` plus the edited files if you made fixes. It prints what it included and the size. Tell the human to send
**`results.zip`** back to whoever gave them this package. Do not send the whole repository.

## 8. Final report to give the human

Reply with a short report using this template, filled from the real files:

```
Hardware: <GPU name>, <memory> MiB, driver <version>; torch <version>, CUDA <version>
Config: <CONFIG>, flags used: <flags>
Timing: smoke test OK/failed; seq2seq <minutes>, <step/s>; RL <minutes>; total <hours>
Result table (from runs/compare.md, the "Overall" rows):
  retrieval baseline / supervised / supervised+RL: 1st-compile, compile+conform (final), success@3
Held-out identifiers row (from "Held-out identifier subset"): <numbers>   <- the most important line
Best checkpoint chosen: <supervised | supervised+RL> (see checkpoints/best/SOURCE.txt)
Problems met and what was done: <none | list>
Code changes: <none | see CHANGES.md>
Deliverable: results.zip (<size> MB)
```

For reference, a small CPU-trained model (the `cpu` config, 4.4M parameters, about one pass over the data, 600 test examples)
reached **73.2%** compile-and-meet-spec overall and **55.0% on held-out identifiers** (names never seen in training; 93.0% on
names seen in training). This run uses a larger model and a bigger time budget, so the aim is to beat those two numbers,
especially the held-out one. Report what you measure exactly, good or bad; if the larger model is not better, say so.

## 9. Quick reference

| thing | where |
|---|---|
| full documentation | `README.md` |
| configs | `mjava/configs/{gpu4,gpu,cpu,tiny}.yaml` |
| stage list and flags | `$PY scripts/run_all.py --help` |
| progress / resume state | `runs/run_all_state.json` |
| trained model (best) | `checkpoints/best/` |
| results table | `runs/compare.md` |
| use the trained model | `$PY -m mjava.cli examples/Student.mjava -o out/Student.java --verbose` (needs `checkpoints/best`) |
