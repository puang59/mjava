# Running mjava locally (Windows first)

Step-by-step guide for an AI agent or a human to get `mjava` running on a fresh machine, written for a **Windows laptop with
PowerShell**. macOS/Linux equivalents are at the end. Follow the steps in order; each has a check that must pass before you
continue.

What you get from this repo out of the box:

- the code (`mjava/`, `scripts/`, `tests/`),
- the **already trained model** in `checkpoints/best/` (~18 MB, CPU is enough to use it),
- example inputs in `examples/` and the results table/plots in `runs/`.

What is **not** in the repo (too large for GitHub): the dataset `data/` (~600 MB), intermediate checkpoints and optimizer
states (`*.pt` other than `checkpoints/best/model.pt`). You only need those to *train*; see step 7.

---

## 1. Install the prerequisites

| need | version | install (PowerShell, one at a time) |
|---|---|---|
| Git | any | `winget install Git.Git` |
| Python | **3.11, 3.12 or 3.13** (not 3.14, torch may lack wheels) | `winget install Python.Python.3.12` |
| JDK (`javac` + `java`) | 11 or newer, **17 recommended** | `winget install EclipseAdoptium.Temurin.17.JDK` |

After installing, **close and reopen PowerShell** so `PATH` is refreshed. Then check:

```powershell
git --version
py -3 --version          # must say 3.11.x, 3.12.x or 3.13.x
javac -version
java -version
```

If `javac` is not found after installing the JDK: the installer did not add it to `PATH`. Find the folder (usually
`C:\Program Files\Eclipse Adoptium\jdk-17.*\bin`) and add it for the current session:

```powershell
$env:Path = "C:\Program Files\Eclipse Adoptium\jdk-17.0.13.11-hotspot\bin;" + $env:Path   # adjust the folder name
```

(or add it permanently via *Settings > System > About > Advanced system settings > Environment Variables*).

If `py` is missing but `python` works, use `python` instead of `py -3` below. If `python` opens the Microsoft Store, Python is
not really installed; install it with `winget` as above.

## 2. Get the code

```powershell
git clone https://github.com/puang59/mjava.git
cd mjava
```

All later commands are run **from this folder** (the one containing `pyproject.toml`). The default model path
`checkpoints\best` is relative to the current directory.

## 3. Create a virtual environment

Call the venv's Python **by path** instead of activating it. This avoids the PowerShell execution-policy error
(`Activate.ps1 cannot be loaded because running scripts is disabled`).

```powershell
py -3 -m venv .venv
$PY = ".venv\Scripts\python.exe"
& $PY -m pip install --upgrade pip
```

`$PY` only lives in the current PowerShell window; set it again if you open a new one.

## 4. Install PyTorch, then the project

**Just using the trained model (CPU is fine):**

```powershell
& $PY -m pip install torch
& $PY -m pip install -e ".[dev]"
```

**Training on an NVIDIA GPU:** plain `pip install torch` on Windows installs a CPU-only build. Instead open
<https://pytorch.org/get-started/locally/>, select Stable / Windows / Pip / Python / a CUDA version your driver supports
(`nvidia-smi` shows the driver), and run the printed command through `& $PY -m pip ...`, e.g.
`& $PY -m pip install torch --index-url https://download.pytorch.org/whl/<tag-from-that-page>`. Then
`& $PY -m pip install -e ".[dev]"`.

Check:

```powershell
& $PY -c "import torch, mjava; print(torch.__version__, torch.cuda.is_available())"
```

Must print a version and `False` (CPU) or `True` (GPU) without errors.

## 5. Run the tests

```powershell
& $PY -m pytest -q -m "not slow"
```

Expected: roughly `141 passed, 1 skipped, 7 deselected` in under a minute. The tests that need a JDK are marked `javac`; if
those fail, step 1's `javac -version` check is not really passing in this shell.

## 6. Use the trained model (`mjavac`)

```powershell
& $PY -m mjava.cli examples\Student.mjava -o out\Student.java --verbose
```

Expected: a few `attempt N` blocks, then `block 1: OK ...` and `mjavac: wrote ...\out\Student.java`, exit code 0
(check with `$LASTEXITCODE`). The first run takes a few extra seconds because it compiles the small Java compile server into
`mjava\compiler\java\build\`.

More examples:

```powershell
& $PY -m mjava.cli examples\Geometry.mjava -o out\Geometry.java --verbose
& $PY -m mjava.cli examples\Library.mjava --max-retries 5 --run     # --run also executes main()
& $PY -m mjava.cli examples\Stress.mjava --verbose                  # hard case; may be flagged for review
```

`.venv\Scripts\mjavac.exe` is the same command (`.venv\Scripts\mjavac examples\Student.mjava`).

Exit codes: `0` compiled and written; `1` still not compiling after the retries (a `<file>.mjava.review.txt` is written next to
the input with the last attempt and compiler errors); `2` usage/environment problem (no checkpoint, no JDK); `3` only with
`--run`, the program compiled but crashed. The `.mjava` input is never modified.

To use a model in another folder: `--checkpoint D:\models\best` or `$env:MJAVA_CHECKPOINT = "D:\models\best"`.

## 7. (Optional) Train or evaluate

Quick end-to-end check of the whole pipeline with a tiny throw-away model (a few minutes on CPU, writes only to
`runs\smoke\` and `checkpoints\smoke\`):

```powershell
& $PY scripts\run_all.py --preset smoke
```

A real training run needs the dataset. `data\` is not in the repo; if `data\meta.json` is missing, `run_all.py` rebuilds it
first (several minutes, ~600 MB). Note: the compiler-error wording in the repair data depends on the JDK version, so build it
with JDK 17 to match the shipped model. Then:

```powershell
& $PY scripts\run_all.py --config cpu --device cpu                  # CPU, ~3 h for the default budgets
& $PY scripts\run_all.py --config gpu                               # NVIDIA GPU (8-12 GB); gpu4 for 4-6 GB cards
```

Training **overwrites `checkpoints\best\`** at the end. Copy it somewhere first if you want to keep the shipped model. For long
GPU runs (background process, monitoring, out-of-memory fixes) follow `START_HERE_AGENT_INSTRUCTIONS.md`; full documentation is
in `README.md`.

## Troubleshooting

| symptom | fix |
|---|---|
| `javac not found: install a JDK ...` | JDK missing or not on `PATH` in this shell (step 1) |
| `Activate.ps1 cannot be loaded` | don't activate; use `& $PY ...` (step 3), or `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `No matching distribution found for torch` | Python version too new/old or 32-bit; use 64-bit Python 3.11-3.13 |
| `checkpoint ... not found` / exit code 2 | run from the repo root, or pass `--checkpoint` |
| `torch.cuda.is_available()` is `False` on a GPU laptop | CPU-only torch; `& $PY -m pip uninstall -y torch` and reinstall with the CUDA index URL (step 4) |
| `ModuleNotFoundError: mjava` | `& $PY -m pip install -e ".[dev]"` was skipped or used a different Python |

## macOS / Linux

Same steps, with these substitutions: install Python 3.11+ and a JDK (`brew install python@3.12 openjdk@17`, or
`sudo apt install python3-venv openjdk-17-jdk`), then

```bash
python3 -m venv .venv
PY=.venv/bin/python
$PY -m pip install torch && $PY -m pip install -e '.[dev]'
$PY -m pytest -q -m "not slow"
$PY -m mjava.cli examples/Student.mjava -o out/Student.java --verbose
```
