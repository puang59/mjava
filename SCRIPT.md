# Presentationcript - `mjava`
### 1. Opening (30 sec)

- One sentence: what the project is.
  > "We built a small Transformer model, trained from scratch, that turns plain
  > English instructions into real, compilable Java code."
- One sentence: what makes it more than "just text generation."
  > "It checks its own output against the real Java compiler, and if it's
  > wrong, it reads the compiler's error message and fixes itself."

### 2. Live demo #1 — the simple case (1 min)

- [Show `examples/Student.mjava` — 3 lines of plain English]
- [Run: `mjavac examples/Student.mjava --verbose`]
- Narrate while it runs: "It's generating Java token by token, then compiling
  it for real before it ever shows us the result."
- [Point at the generated Java, then `javac` it live to prove it compiles]
- Line to land: "That's the whole idea — English in, compiling Java out."

### 3. One-line architecture + training (1 min, don't over-explain)

- "Under the hood it's an encoder-decoder Transformer — the same family of
  model behind things like GPT — about 4.4 million parameters."
- "Training happens in two stages: first supervised, on 300,000 example
  pairs; then reinforcement learning, where the reward is literally 'did it
  compile and did it match the spec.'"
- Keep this short — save depth for if they ask.

### 4. Live demo #2 — the harder case (2 min, the centerpiece)

- [Show `examples/Stress.mjava` — inheritance, interface, abstract class, enum]
- [Run with `--verbose`]
- Narrate: "Watch block 3 — it fails the first attempt, reads the real
  compiler error, and tries again."
- Let the retries actually print; don't skip past them.
- Land it honestly: "It doesn't always succeed — here it struggles with two
  similarly-named fields — but it fails *safely*: it tells us exactly why and
  flags it for review instead of pretending it worked."

### 5. The numbers (1 min)

- [Show `runs/compare.md` or just say the numbers]
- "After the correction loop, the model compiles and matches the spec about
  73% of the time, up from about 7% for a naive retrieval baseline."
- The one number to really emphasize:
  > "On names it's never seen before, it drops to 55%, versus 93% on names
  > it has seen. That gap is the real story — it learned the *structure* of
  > the task, but copying brand-new names is still hard. That's the
  > generalization-vs-memorization line from the course, measured directly."

### 6. Honest limitations (30-45 sec)

- "It's also brittle to phrasing — short, casual instructions that drift from
  the training style break it much faster than just inventing weird names
  does."
- [Optional: one quick example, e.g. the "Dog" case where it compiled but
  didn't actually print anything — "compiles ≠ correct"]

### 7. Close (15-20 sec)

- One sentence wrap-up: what you'd do next if you kept going (bigger model,
  more phrasing variety in training data, etc.) — pick one, don't list five.
- "Happy to answer questions or show anything else live."

---

## Quick command cheat-sheet for the demo

```
mjavac examples/Student.mjava --verbose     # simple, clean success
mjavac examples/Stress.mjava --verbose      # harder, shows retries + one honest failure
cd out && javac Student.java                # prove it really compiles
```

## If she asks something you don't want to improvise on

Point to `PRESENT_TO_PROFESSOR.md` in this repo — it has the detailed,
file-and-line-referenced answers (architecture, RL algorithm, what's used vs.
not from the syllabus) without you needing to recall it live.
