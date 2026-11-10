# Presenting `mjava` 

This file maps your course syllabus onto what this project *actually does*, with
exact file/line references so you can show your professor the real code, not just
describe it. Every number below comes from a real file in this repo or a real
training log — nothing is invented. Where the project does **not** use a syllabus
topic, that is said plainly, because an honest "we didn't use this, and here's
why" is more convincing than pretending.

---

## 1. The project in one paragraph

`mjava` is a small Transformer (encoder-decoder, built from scratch in plain
PyTorch — no Hugging Face model classes) that translates a plain-English
instruction like *"Create a class Student with fields: name as String, age as
int"* into Java source code, then checks its own output against the real `javac`
compiler. If the code doesn't compile, the model gets the exact compiler error
fed back to it and retries (up to 3 times). It is trained in two stages: first
with **supervised learning** (show it correct English → Java pairs), then with
**reinforcement learning** where the reward signal comes from whether the code
actually compiles and matches the spec. That second stage is the part of this
project that lines up most directly with your Module 7.

Model file: `mjava/model/transformer.py`. Training: `mjava/train/supervised.py`
(supervised stage) and `mjava/train/rl.py` (RL stage). Self-correction loop:
`mjava/pipeline/correct.py`.

---

## 2. Quick map: your syllabus vs. this project

| Module | Topic                                                                                                 | Used here?                                                                                                               |
| ------ | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| 1      | NN basics, activation/loss functions, forward/back-prop, hyperparameters                              | **Yes** — core of everything                                                                                             |
| 2      | CNNs (ResNet, AlexNet, InceptionNet)                                                                  | **No** — no images, nothing to convolve over                                                                             |
| 3      | Mini-batch GD, Adam, hyperparameter tuning, batch norm, softmax, data augmentation, over/underfitting | **Yes**, except batch norm (see §3.3)                                                                                    |
| 4      | RNNs, encoder-decoder, seq2seq, BERT                                                                  | **Partially** — encoder-decoder/seq2seq yes, but built with attention, not recurrence; no RNN cells; not BERT (see §3.4) |
| 5      | Long-term dependencies, LSTM/gated RNNs, explicit memory                                              | **No RNNs used**, but the project's whole design is a direct answer to this module's problem (see §3.5)                  |
| 6      | Transfer learning, GANs, R-CNN, YOLO, SSD                                                             | **No** — all computer-vision detection topics; one *adjacent* idea is used (see §4)                                      |
| 7      | Deep RL: Q-learning, policy gradients, A2C/A3C                                                        | **Yes, specifically policy gradients** — not Q-learning (see §3.6)                                                       |

---

## 3. What's actually used, in depth

### 3.1 Module 1 — Neural network basics

**Where:** `mjava/model/transformer.py`
<!--  -->
- **Forward propagation**: `Transformer.forward()` (line 246) — takes a batch of
  token ids, pushes them through embeddings → encoder layers → decoder layers →
  a final linear layer that scores every possible next word. This *is* forward
  propagation; you can show your professor this exact function.
- **Backpropagation**: never written by hand — it's `loss.backward()` in
  `mjava/train/supervised.py` line 250. PyTorch's autograd computes every
  gradient automatically. Worth saying explicitly to your professor: the project
  demonstrates understanding of *architecture*, and relies on the framework for
  the calculus, which is the normal division of labor in real deep learning work.
- **Activation function**: GELU, not the classic ReLU or sigmoid. Used inside
  every feed-forward block (`FeedForward.forward`, line 115-116):
  `w2(dropout(gelu(w1(x))))`. In simple words: GELU is a smoothed version of
  ReLU — instead of a hard cutoff at zero, it lets a little bit of the negative
  input through, weighted by how "negative" it is. Modern Transformers (GPT, BERT)
  use GELU instead of ReLU because it tends to train slightly better in practice.
- **Loss function**: cross-entropy, computed once per training step
  (`supervised.py` line 245-248):
  ```
  loss = -log( P(correct next token) )
  ```
  averaged over every real (non-padding) token in the batch. In simple words:
  at every position, the model outputs a probability for every possible next
  token in its 6,000-word vocabulary; the loss punishes it in proportion to how
  low a probability it assigned to the token that was *actually* correct. Get it
  exactly right → loss near 0. Get it very wrong → loss shoots up.
- **Hyperparameters vs. parameters** — a concrete, defensible answer if asked:
  - *Parameters* (learned by gradient descent): the actual trained weights. The
    checkpoint you'll demo has **4,427,520 of them** (confirmed by loading
    `checkpoints/best` and calling `model.num_parameters()`).
  - *Hyperparameters* (chosen by you, not learned): embedding size `d_model=192`,
    number of attention heads `n_heads=4`, number of encoder/decoder layers
    `n_enc_layers=n_dec_layers=3`, feed-forward width `d_ff=768`, `dropout=0.1`,
    vocabulary size `6000`, learning rate, batch size, etc. All in
    `mjava/configs/cpu.yaml` / `gpu.yaml`.

### 3.2 Function approximation / classification framing

Although the output is Java source code, at every single decoding step the
model is solving a **classification problem over 6,000 classes** (one class per
vocabulary token) — exactly the "classification problem" framing from Module 1.
`mjava/model/decoding.py` line 70-79 (`process_logits`) shows the raw scores
being turned into a probability distribution with `log_softmax`
(`decoding.py` line 99). Generating a whole Java file is just this one classification
step, repeated hundreds of times, each time conditioned on everything generated
so far.

### 3.3 Module 3 — Improving deep neural networks

**Where:** `mjava/train/supervised.py`, `mjava/train/rl.py`

| Syllabus topic | In this project |
|---|---|
| Mini-batch gradient descent | `make_batches()` groups examples by a *token budget* (`batch_tokens`), not a fixed example count, because Java snippets vary wildly in length — a batching detail worth mentioning, it shows you understand *why* naive fixed-size batching is wasteful for variable-length text. |
| Adam / RMSProp-style optimization | `torch.optim.AdamW(..., betas=(0.9, 0.98))` — `supervised.py` line 163-167 and `rl.py` line 628-631. In simple words: Adam keeps a running average of each weight's past gradients (momentum) *and* a running average of how big those gradients have been (so it can shrink the step size for weights that are jumping around a lot, and allow bigger steps for calm ones). AdamW additionally separates out weight decay (see below) so it doesn't interact badly with Adam's adaptive step sizes. |
| Hyperparameter tuning | Three tuned configs exist (`mjava/configs/{cpu,gpu,gpu4}.yaml`) with different learning rates, warmup steps, and batch sizes for different hardware — a direct, visible artifact of hyperparameter search. |
| Learning-rate schedule | Warmup (linear ramp for the first `warmup_steps` steps) then cosine decay, see `lr_at()` in `mjava/train/common.py`. The *why*: starting at full learning rate on a randomly-initialized model causes unstable early updates; warmup avoids that, then cosine decay lets the model settle into a minimum smoothly as training ends. |
| **Batch Normalization** | **Not used.** The model uses `nn.LayerNorm` instead (`transformer.py` lines 122,124,136,138,140,196,197). Good thing to explain rather than hide: BatchNorm normalizes across a batch of examples, which works well for images but is awkward for sequences of different lengths (what do you average over when sentences have different lengths and there's padding?). LayerNorm instead normalizes *across the features of a single token*, independent of batch size or sequence position — which is why essentially every Transformer (GPT, BERT, this project) uses LayerNorm, not BatchNorm. This is a great "I understand the tradeoff, not just the keyword" answer if asked. |
| Softmax / softmax classifier | Two places: (1) inside every attention head, `softmax(QKᵀ/√d)V` — see §3.4 math below; (2) at the very output, scoring the next token over the 6,000-word vocabulary (`log_softmax`, `decoding.py`). |
| Data augmentation | `mjava/data/mutate.py` — **this is real, used data augmentation.** It takes correct, compiling Java and deliberately breaks it (drops a semicolon, drops/adds a brace, drops a parenthesis, corrupts a variable name) to synthetically manufacture realistic *broken* Java paired with the real compiler error it produces. These broken→fixed pairs become the "repair" training examples (90,000 of them, out of 300,000 total training examples — see `data/meta.json`) that teach the model to read a compiler error and fix code. This is the single most natural, defensible "data augmentation" story you can tell: it's not augmentation for augmentation's sake, it's manufacturing exactly the failure mode (compile errors) the deployed system needs to learn to recover from. |
| Underfitting vs. overfitting | Real numbers, from `train_run.log` (not cherry-picked, this is the literal training log of the checkpoint you'll demo): validation loss fell steadily from **2.39 → 1.06 → 0.77 → 0.59 → 0.48 → ... → 0.067** over the supervised stage, then **flattened out** around step 19,500–20,700 (0.0673 → 0.0674 → 0.0679, no further improvement). That flattening is the signal training was stopped at the right point: the code picks the checkpoint with the *lowest validation loss seen so far* (`supervised.py` line 213-216, `improved = r["loss"] < best_val`), not just whatever the last checkpoint happens to be — a simple form of early stopping. Regularizers fighting overfitting the whole time: `dropout=0.1` throughout the network, `label_smoothing=0.1` on the loss (softens the targets so the model isn't pushed to be *infinitely* confident), and `weight_decay=0.01` in AdamW (shrinks weights toward zero unless the data insists otherwise). |

### 3.4 Module 4 — Encoder-decoder / sequence-to-sequence (with an honest caveat)

**Where:** `mjava/model/transformer.py`

The project **is** an encoder-decoder sequence-to-sequence model — the encoder
reads the English instruction, the decoder generates Java token-by-token while
"looking back" at the encoder's output. That's `Transformer.encode()` (line 231)
and `Transformer.forward()`/`decode_step()`.

**The honest caveat:** your syllabus's Module 4 teaches this concept using
**RNNs** (recurrent cells stepping through a sequence one token at a time,
carrying a hidden state forward). This project's encoder-decoder uses **no RNN
cells at all** — the "encoder" and "decoder" are both stacks of **attention +
feed-forward layers**, the Transformer architecture. Say this straightforwardly
to your professor — it shows you know the difference between the *concept*
(encoder-decoder/seq2seq) and the *mechanism* (RNN vs. attention) your syllabus
happens to teach it with.

**The math, in simple words** (`MultiHeadAttention.attend`, line 80-86):
```
Attention(Q, K, V) = softmax( Q·Kᵀ / √d_head ) · V
```
- Every token produces a **Query** (what am I looking for?), a **Key** (what do I
  contain?), and a **Value** (what do I actually offer if picked?).
- `Q·Kᵀ` is a similarity score between every pair of tokens — how much should
  token *i* pay attention to token *j*?
- Dividing by `√d_head` just keeps those scores from growing too large as the
  vector size grows (otherwise softmax saturates and gradients vanish).
- `softmax(...)` turns the raw scores into a probability distribution — "pay
  60% attention to token 3, 30% to token 7, 10% to everything else."
- Multiplying by `V` produces a weighted blend of every token's Value vector,
  weighted by that attention distribution.
- "Multi-head" (`n_heads=4`) just means this whole thing is done 4 times in
  parallel with different learned projections, so the model can track 4
  different kinds of relationships at once (e.g., one head might track "which
  field name goes with which type," another might track "which method matches
  which instruction clause").

**BERT**: mentioned in your syllabus, **not used here**. BERT is an
*encoder-only* model pretrained by predicting randomly *masked* words using
context from both directions. This project's optional pretraining stage
(`build_lm_pairs`, `mjava/train/common.py` line 146) is the opposite: a
**decoder-only, left-to-right** "predict the next token" pretraining pass over
raw Java code — that's closer to how GPT is pretrained than how BERT is. Good
thing to be precise about if asked.

### 3.5 Module 5 — Long-term dependencies (not used, but directly answered)

Your syllabus's Module 5 is about why plain RNNs struggle to remember
information from far back in a sequence (vanishing gradients through many
recurrent steps), and the fixes: LSTMs, gated RNNs, "explicit memory."

This project uses **none of those** — no LSTM, no GRU, no Echo State Network.
Instead, it sidesteps the whole problem: attention lets every decoder step look
directly at every encoder token in one matrix multiply, with no chain of
recurrent steps to vanish through. That *is* the historical reason Transformers
replaced RNNs for this kind of task, and it's worth saying so explicitly — the
project's architecture choice is a direct answer to the problem Module 5
describes, even though it doesn't use any of Module 5's named techniques.

One genuine, concrete parallel to "explicit memory": the incremental decoder
keeps a **KV cache** (`DecoderState`, `transformer.py` line 155-183) — literally
a bank of previously computed Key/Value vectors it reads from at every new
step instead of recomputing everything. It's not the "explicit memory" your
syllabus means (e.g., Neural Turing Machines), but if your professor asks "do
you have anything like external memory," this is a fair, honest thing to point
at.

### 3.6 Module 7 — Deep reinforcement learning (the strongest match)

**Where:** `mjava/train/rl.py` (read the module docstring, lines 1-43 — it's
genuinely one of the clearest descriptions of the algorithm you'll find)

This is the part of the syllabus this project engages with most seriously, so
it's worth spending real presentation time here.
<!--  -->
**What's used: policy gradients (REINFORCE), self-critical sequence training.**
**What's *not* used: Q-learning / Deep Q-learning, and *not* true Actor-Critic**
(there is no learned value/critic network — see below). Say this distinction
out loud; it's the single most impressive thing you can say in this section,
because it shows you know RL well enough to tell the difference.

**The algorithm, in simple words** (this is literally what `rl_step()` does,
`rl.py` line 372-431):

1. For a batch of instructions, decode the Java **two different ways**:
   - **Greedy** (always pick the single most likely next token) → this is the
     **baseline**, not a separate neural network, just the same model doing its
     "safest" decode.
   - **Sampled** (pick randomly from a temperature-softened, top-k-filtered
     distribution, 2 samples per instruction) → these are the "exploratory"
     attempts.
2. Compile *both* the greedy and the sampled outputs with the real `javac`
   compiler and score them with a **reward**:
   ```
   reward = conformance score,                      if it compiles
   reward = 0.3 × conformance × (1 − 0.2 × n_errors), if it doesn't compile
   reward = 0,                                        if it crashes/times out/truncates
   ```
   ("conformance" = does the generated class actually have the fields/methods the
   instruction asked for, checked with `mjava/data/conformance.py`.) This reward
   is **not** differentiable — you can't backpropagate through "did `javac`
   accept this file?" — which is exactly why reinforcement learning, not plain
   supervised learning, is needed here.
3. Compute the **advantage** for each sampled output:
   ```
   advantage = reward(sampled attempt) − reward(greedy attempt on the same input)
   ```
   In simple words: "was this risky, randomly-sampled attempt better or worse
   than just playing it safe?" A positive advantage means "do more of
   whatever made this sample different"; negative means "do less of that."
4. Turn the advantage into a loss the optimizer can use
   (`policy_loss`, line 328-346):
   ```
   loss_RL = − mean( advantage × average log-probability the model assigned to its own sampled tokens )
   ```
   In simple words: if a sample scored *better* than the greedy baseline,
   increase the probability the model would generate exactly that sequence
   again. If it scored *worse*, decrease that probability. This is the
   classic REINFORCE / policy-gradient update rule.
5. This is blended with an ordinary supervised cross-entropy loss on the same
   batch:
   ```
   total loss = 0.5 × loss_RL + 0.5 × loss_supervised
   ```
   (`lambda_rl = 0.5`, `rl.py` line 87) — pure RL from scratch is unstable and
   sample-inefficient for something as structured as code, so the supervised
   loss acts as an anchor that keeps the model from drifting into gibberish
   while it explores.

**Why this is "self-critical sequence training" (SCST) and not Actor-Critic
(A2C):** A2C trains a second, separate network (the critic) whose whole job is
to *predict* how good a state is, and uses that prediction as the baseline.
Here, there is no critic network — the "baseline" is just the same policy's own
greedy decode, scored by the real compiler, recomputed fresh every batch. It's
cheaper and simpler than A2C, at the cost of being noisier and less sample
efficient; a reasonable engineering tradeoff for a reward (compiling code) that
is itself cheap to compute. If your professor pushes on this, the honest and
correct answer is: *"We use REINFORCE with a self-critical (self-generated)
baseline, which is a simpler, older cousin of actor-critic methods — same idea
of 'was this better than a baseline,' but the baseline comes from the policy
itself instead of a learned critic network."*

**Real numbers from an actual RL run** (`train_run.log`, tail of the file —
this is the RL fine-tuning stage that started from the already-trained
supervised checkpoint):
```
step  50: compile 0.929, conformance 0.984, fully-correct 0.900
step 100: compile 0.936, conformance 0.985, fully-correct 0.900  (best, saved)
step 150: compile 0.936, conformance 0.986, fully-correct 0.907  (best, saved)
step 200: compile 0.943, conformance 0.986, fully-correct 0.907  (best, saved)
```
Small but real, monotonic improvement in exactly the metric the reward was
built to reward — good evidence the RL stage is doing something, not just
noise.

---

## 4. The one idea from Module 6 that's loosely relevant: transfer learning

Everything else in Module 6 (GANs, R-CNN, Fast R-CNN, YOLO, SSD) is computer
vision object-detection/generation — genuinely **not applicable**, there are no
images anywhere in this project. Say so plainly if asked.

One *adjacent* idea is worth mentioning: the RL stage never trains from a
random initialization. It always starts from the already-trained supervised
checkpoint (`--init-from`, `rl.py` lines 578-584, `model, tok = load_checkpoint(model_dir, ...)`)
and *fine-tunes* those same weights with a new objective (the RL loss) on top.
That is the general definition of transfer learning / fine-tuning — just
within the same task and domain, rather than the cross-domain case (e.g.
ImageNet → medical images) your syllabus's Module 6 is really describing.
Useful to mention, but don't oversell it as "we did transfer learning" in the
Module 6 sense.

---

## 5. Headline numbers to have ready (all from `runs/compare.md`, a real,
generated results table — not hand-written)

```
system              | 1st-try compiles | compiles AND matches spec (after retries)
retrieval baseline   |      85.8%       |   7.3%
supervised only      |      83.7%       |  73.0%
supervised + RL      |      83.7%       |  73.2%

Held-out identifiers (names never seen during training) vs. seen identifiers:
  seen names:      93.0% compile+conform
  held-out names:  55.0% compile+conform
```

That gap (93.0% vs. 55.0%) is the single most important number in the whole
project for a deep learning class: it's direct, measured evidence of the
difference between **memorization** and **generalization** — the model
clearly learned the *structure* of the task, but still struggles to precisely
*copy* novel names it has never seen. If you did nothing else quantitative in
the presentation, show this one row.

---

## 6. Suggested presentation structure

1. **One slide, the one-paragraph summary** (§1).
2. **Live demo** (you already have `mjavac` set up as a command — see your
   earlier conversation): run `mjavac examples/Student.mjava --verbose`, show
   the generated Java, compile it with `javac`. Takes 30 seconds, proves the
   whole pipeline is real.
3. **Live demo, harder case**: run a more complex `.mjava` file (you already
   have `examples/Stress.mjava` from testing this) and watch the self-correction
   loop actually retry against real compiler errors — tie this directly to
   the architecture discussion in §3.6 and the held-out-identifier number in §5.
4. **Architecture slide**: the encoder-decoder diagram + the attention formula
   from §3.4. This is where most of your Module 4 and Module 1 points land.
5. **Training slide**: the loss curve numbers from §3.3, label the supervised
   vs. RL stages, point at the validation-loss plateau as your overfitting
   discussion.
6. **RL slide**: walk through the 5-step algorithm in §3.6 in your own words;
   this is your strongest, most specific Module 7 material.
7. **Honest limitations slide**: the 93.0% vs. 55.0% identifier gap (§5), and
   if you tested them, the short-input failures (casual phrasing breaking the
   model) from earlier testing — both are genuine findings, not excuses.

---

## 7. Likely questions and grounded answers

- **"Why Transformers and not RNNs/LSTMs, since that's what we covered?"**
  → §3.5: attention avoids the vanishing-gradient-over-many-steps problem RNNs
  have, by letting every position attend directly to every other position
  instead of passing information through a chain of recurrent steps.

- **"Where's your validation/test split, how do you know it's not just
  memorizing?"** → `data/meta.json` has fixed train/val/test splits (300k /
  3k / 3k generate examples, plus repair examples); the held-out-identifier
  subset in `runs/compare.md` is specifically constructed with names that never
  appear in training, precisely to test memorization vs. generalization (§5).

- **"What's the loss function and why cross-entropy?"** → §3.1: next-token
  classification over a 6,000-word vocabulary at every position; cross-entropy
  is the standard loss for classification because it directly penalizes
  low assigned probability to the correct class.

- **"Is this actor-critic?"** → No — see §3.6's explicit distinction. It's
  REINFORCE with a self-generated (greedy) baseline, not a learned critic.

- **"Did you use batch normalization?"** → No, LayerNorm — and explain why
  (§3.3), it's a stronger, more specific answer than "we didn't need it."
