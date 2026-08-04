# Dravidian-LM: Results

Status snapshot of all evaluation runs completed so far. Source files live in `results/raw/`.
Sections marked **(unverified)** are transcribed from console output pasted into chat, not yet
confirmed against a pushed/pulled JSON file — treat as provisional until the underlying file is
checked in.

## 1. Perplexity (full held-out test split, no subsampling)

Source: `results/raw/{mono,multi}_{language}_perplexity_full.json`

| Model | Language | Eval loss | Perplexity | BPB | Test tokens |
|---|---|---:|---:|---:|---:|
| Monolingual | Telugu | 3.7457 | 42.34 | 0.3733 | 5,119,128 |
| Monolingual | Tamil | 3.9279 | 50.80 | 0.3586 | 18,764,494 |
| Monolingual | Kannada | 3.9537 | 52.13 | 0.3820 | 3,571,107 |
| Monolingual | Malayalam | 3.4223 | 30.64 | 0.3151 | 7,178,537 |
| Multilingual | Telugu | 3.6813 | 39.70 | 0.4092 | 5,708,641 |
| Multilingual | Tamil | 3.7838 | 43.98 | 0.3852 | 20,926,176 |
| Multilingual | Kannada | 3.9777 | 53.39 | 0.4295 | 3,990,425 |
| Multilingual | Malayalam | 3.6324 | 37.81 | 0.3610 | 7,747,667 |

Notes:
- Multi vs. mono perplexity is split, not uniform: the multilingual model has **lower**
  (better) perplexity than its monolingual counterpart for Telugu (39.70 vs. 42.34) and Tamil
  (43.98 vs. 50.80), but **higher** (worse) perplexity for Kannada (53.39 vs. 52.13) and
  Malayalam (37.81 vs. 30.64 — the largest relative gap of the four). This does **not** track
  corpus size cleanly: by test-split file size (a proxy for corpus size, fixed 96/2/2 ratio),
  the ranking is Tamil (298MB) > Malayalam (113MB) > Telugu (74.6MB) > Kannada (53.7MB) — yet
  Telugu (third-largest) gains the most from the multilingual model, while Malayalam
  (second-largest) loses the most. Whatever's driving the split, it isn't simply "smaller
  corpora benefit more from shared capacity." Needs actual investigation before claiming a
  cause in the paper — don't guess at one to fill the gap.
- Token counts differ between mono/multi for the same language because they use different
  tokenizers (32K per-language vs. 64K joint vocabulary), so BPB (bytes/bit, tokenizer-agnostic)
  is the fairer cross-model comparison — and by BPB the ranking flips: monolingual is *better*
  bits-per-byte for every language (e.g. Telugu 0.3733 mono vs 0.4092 multi). The joint
  tokenizer produces fewer, "cheaper" tokens per byte, which flatters raw perplexity while BPB
  corrects for it.

## 2. Tokenizer efficiency

Source: `results/raw/{language}_full_eval.json` (`tokenizer_analysis` key), all four languages.
Sample: 2,000 test-split sentences per language.

| Language | Ours vocab | Ours fertility | Ours compression | Multi vocab | Multi fertility | Multi compression | XLM-R fert. | mBERT fert. | mGPT fert. | mGPT compression |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Telugu | 32,000 | 1.61 | 13.12 | 64,000 | 1.77 | 11.92 | 2.37 | 3.88 | 6.19 | 3.41 |
| Tamil | 32,000 | 1.66 | 14.80 | 64,000 | 1.85 | 13.33 | 2.42 | 3.73 | 6.98 | 3.53 |
| Kannada | 32,000 | 1.59 | 13.60 | 64,000 | 1.77 | 12.27 | 2.40 | 3.95 | **16.39** | **1.32** |
| Malayalam | 32,000 | 1.84 | 14.46 | 64,000 | 1.97 | 13.49 | 2.58 | 5.02 | 7.87 | 3.38 |

Our dedicated per-language tokenizer beats every multilingual baseline on every language, despite
a much smaller vocabulary (32K vs. 100K–250K) — consistently the lowest fertility and highest
compression, with ~0% UNK rate everywhere (mBERT is the only baseline with any UNKs, 0.2–0.3%).
**mGPT's Kannada tokenization is a clear outlier** — 16.39 tokens/word and 1.32 bytes/token, 2–2.5×
worse than mGPT's own numbers on the other three Dravidian languages — suggesting mGPT's
tokenizer has little to no real Kannada-script coverage and is falling back to heavy
byte-fragmentation. Worth citing explicitly as a concrete example of why dedicated tokenization
matters, not just an aggregate stat.

The multilingual model's 64K joint tokenizer is consistently *worse* than each language's
dedicated 32K tokenizer (higher fertility, lower compression, on every language) despite having
twice the vocabulary — direct evidence that shared-vocabulary tokenization costs each individual
language something, even before any model-capacity effects. It's still far better than any of the
general-purpose multilingual baselines, though, since it was at least trained on this language
family specifically rather than on hundreds of unrelated languages.

## 3. Downstream fine-tuning

Full fine-tuning, 5 epochs, IndicSentiment (accuracy) + WikiANN NER (span F1), vs. mGPT baseline
fine-tuned identically. Source: `results/raw/{language}_full_eval.json` (`downstream` key) — this
supersedes an earlier version of this table built from a pasted console summary, which showed
**substantially different numbers for Tamil, Kannada, and Malayalam** (Telugu was unchanged and
matches). See the variance note below before treating any of this as stable.

| Language | Model | IndicSentiment acc. | mGPT acc. | NER F1 | mGPT F1 | NER train n |
|---|---|---:|---:|---:|---:|---:|
| Telugu | Monolingual | **0.7083** | 0.6250 | **0.6092** | 0.3039 | 1,000 |
| Telugu | Multilingual | 0.6667 | 0.5417 | 0.5605 | 0.2699 | 1,000 |
| Tamil | Monolingual | **0.7917** | 0.4583 | **0.6411** | 0.3838 | 15,000 |
| Tamil | Multilingual | **0.7917** | 0.4583 | **0.6102** | 0.3853 | 15,000 |
| Kannada | Monolingual | **0.7917** | 0.5417 | **0.2677** | 0.0341 | 100 |
| Kannada | Multilingual | 0.6250 | 0.6250 (tie) | **0.1933** | 0.0725 | 100 |
| Malayalam | Monolingual | **0.8750** | 0.4583 | **0.6579** | 0.3331 | 10,000 |
| Malayalam | Multilingual | **0.7500** | 0.5000 | **0.6356** | 0.3326 | 10,000 |

IndicXNLI was attempted but is permanently out of scope: `ai4bharat/IndicXNLI` uses a deprecated
Hub loading script, and none of our four languages are covered by the `facebook/xnli` fallback.

Notes:
- **This run: our monolingual model beats mGPT on every task for every language.** That's a much
  cleaner story than the earlier console-paste version of this table (where Tamil and Kannada lost
  to mGPT on both tasks) — and the fact that *both* stories came from nominally the same
  `--seed 42` run is itself the important finding, not either individual result: fine-tuning on
  these dataset sizes is not reproducible run-to-run. Sources of non-determinism: task execution
  order affects how far the RNG has advanced before fine-tuning starts, plus standard CUDA/cuDNN
  non-determinism.
- **IndicSentiment (n=24 test) and Kannada NER (n=100 train / n=100 test — WikiANN's Kannada
  split is far smaller than the other three languages') are both too small to trust a single run
  of.** No additional seeds are available (compute/time constraint), so this table is being
  reported single-run, with an explicit "not averaged, indicative only" caveat wherever it
  appears in the paper — not silently presented as a stable point estimate.
- NER F1 at n=1,000 (Telugu, Tamil, Malayalam) is comparatively more trustworthy than Kannada's
  n=100 or IndicSentiment's n=24, but "more trustworthy" here is relative, not absolute.
- **Multi model downstream scores are now complete for all four languages.** Pattern: multi
  matches or trails mono on most language/task pairs (e.g. Malayalam sentiment 0.75 vs. mono's
  0.875; Kannada NER 0.1933 vs. mono's 0.2677), except Tamil sentiment where they tie at 0.7917.
  Kannada is the one case where multi's IndicSentiment score exactly **ties mGPT** (0.6250 vs.
  0.6250) rather than beating it — the only language/task pair in this entire table where our
  model doesn't outright win. Given the multi model is the one with the training divergence
  (Section 4) and is effectively only epoch-1-trained, "usually a bit behind mono, still ahead of
  mGPT on most tasks" is a reasonable reading — but this table carries the same single-run caveat
  as everything else here.

## 4. Pretraining convergence (validation split, training-time — not the Section 1 test-split numbers)

Two sources, covering different models:

**Local seed logs** (`results/raw/{language}_seed{n}.json`) — final-epoch numbers only:

| Language | Seed | Eval loss | Perplexity | Train runtime |
|---|---|---:|---:|---:|
| Telugu | 1 | 3.7620 | 43.03 | 241,132s (~67.0h) |
| Telugu | 2 | 3.7635 | 43.10 | 241,547s (~67.1h) |
| Kannada | 1 | 3.9794 | 53.49 | 181,630s (~50.5h) |

Seed variance is small for Telugu (43.03 vs. 43.10), a reasonable sanity check that training is
stable *for that model*. All runs: 4×GPU via Accelerate, 3 epochs, effective batch 256, LR 1e-4
cosine, bf16/tf32.

**Full per-step curves**, extracted from each model's final HF Hub checkpoint's
`trainer_state.json` (`log_history` is cumulative — the last checkpoint alone has the whole run's
training-loss-every-200-steps and eval-loss-every-epoch history; see
`src/dravidian_lm/analysis/plot_training_curves.py`, output in `results/raw/training_curves.json`
and `results/figures/{telugu,tamil,malayalam,multi}_training_curve.{png,pdf}`, one figure per
model). Covers Telugu, Tamil, Malayalam, and the multilingual model —
**Kannada has no checkpoint subfolders on the Hub, and the local checkpoint has since been
deleted from the training cluster, so this gap is permanent**: no training curve is recoverable
for Kannada, only the single final data point in the local-seed-log table above.

| Model | Eval epoch(s) | Eval loss | Eval perplexity |
|---|---|---|---|
| Telugu (mono) | 1.0 / 2.0 / 3.0 | 4.0202 / 3.8219 / 3.7635 | 55.7 / 45.7 / 43.1 |
| Tamil (mono) | 1.0 / 2.0 | 4.0392 / 3.9189 | 56.7 / 50.3 |
| Malayalam (mono) | 1.0 / 2.0 / 3.0 | 3.7881 / 3.5708 / 3.5097 | 44.1 / 35.5 / 33.4 |
| Multilingual | 1.0 / 2.0 | 3.7641 / **8.7017** | 43.1 / **6011** |

### Training instability in the multilingual model

The multilingual run **diverged mid-training and never recovered**. Training loss was healthy
(~3.5–4.7) through step ~766,000, then spiked sharply between steps 766,000–766,400 (loss 4.71 →
6.70 → 9.21), had a few more spike/partial-recovery cycles through ~step 783,600, then plateaued
around loss 8.6–8.7 for the rest of training through the final step (1,078,248 — the end of the
nominal 3rd epoch). In epoch terms, this is roughly 71% through the full 3-epoch schedule, i.e.
partway through epoch 2 of 3.

This also resolves an apparent contradiction: the epoch-2 eval_loss recorded in this checkpoint's
history is 8.70 (perplexity ≈6,000, reflecting the diverged state), yet the root-level model on
the Hub — the one every perplexity/downstream number in this document actually used — scores
eval_loss ≈3.68–3.98 across languages (Section 1), consistent with the *epoch-1* eval_loss of
3.764. `train.py` sets `load_best_model_at_end=True` with `metric_for_best_model="eval_loss"`, so
the Trainer reloaded the best (epoch-1) checkpoint before the final `save_model()` call that
produced what's on the Hub. **In effect, the released `dravidian-gpt2-multi` model is an
epoch-1-equivalent checkpoint, not a fully 3-epoch-trained model** — roughly a third of its
nominal training budget before things went wrong, versus the monolingual models, which all
completed clean 3-epoch runs.

This changes how Section 1's multi-vs-mono perplexity split should be read: multi beats mono on
Telugu and Tamil, and loses on Kannada and Malayalam — and it does so as an *undertrained*
model relative to the mono baselines. If anything, multi winning on 2 of 4 languages despite
effectively 1/3 the completed training is a more interesting result than a clean win, not a
weaker one — but it also means the multi-vs-mono comparison in this paper is not
apples-to-apples on training budget, and that should be stated explicitly as a limitation rather
than glossed over.

**Root cause: gradient explosion.** `grad_norm` (logged pre-clipping) was already elevated for
hundreds of steps before the spike — 100–550 at steps 765,000–766,000, versus a configured
`max_grad_norm=0.5` clip target — then exploded at the divergence point itself: step 766,400 hit
`grad_norm=18,979`, followed by 29,245 and 32,450 over the next few hundred steps, coinciding
exactly with the loss jump from 4.71 to 9.21. Gradient clipping rescales the update direction to
norm ≤0.5 but does not protect Adam's running second-moment estimate from a gradient that
extreme — consistent with the loss plateauing afterward rather than recovering, i.e. optimizer
state corruption, not just one bad step. Plausible contributors not yet isolated: bf16 numerical
range over a single uninterrupted cosine schedule, and/or the larger 64K-vocab embedding table
(135M params vs. 110M) making this model more sensitive than the four monolingual runs, none of
which show any comparable spike. Retraining is presumably out of scope now that the model is
released and evaluated, but the limitation needs to be disclosed in the paper regardless.

## 5. Known caveats affecting interpretation

- **IndicSentiment test set is tiny (~24 examples)** — see Section 3.
- **Perplexity uses fixed-length truncation (1024 tokens), not sliding-window**, despite an
  inaccurate docstring in `perplexity.py` claiming otherwise — long test lines lose content past
  the first 1024 tokens rather than being scored in overlapping windows.
- **The `{lang}_train_balanced.txt` / `{lang}_curriculum_ordered_balanced.txt` files on the HF
  dataset repo don't correspond to any script in this codebase** — `split.py` only ever produces
  a plain `{lang}_train.txt`. Unresolved: confirm what actually produced the balanced/curriculum
  files and whether that's what training actually consumed.

## 6. Outstanding work

- [x] Tokenizer efficiency: all four languages done and saved (Section 2)
- [x] Downstream: all four monolingual `_full_eval.json` files pushed/pulled and correctly
      attributed (Section 3)
- [x] **Downstream reproducibility — decided: no additional seeds available (compute/time
      constraint), so this is being handled as a disclosed limitation, not fixed by
      averaging.** Section 3's numbers are single-run and must be reported as such in the
      paper — every downstream table/figure needs an explicit "single seed, not averaged"
      caveat, and the Limitations section should state plainly that the two runs observed for
      the same nominal seed disagreed substantially for Tamil/Kannada/Malayalam, so these
      numbers should be read as indicative, not as a stable point estimate.
- [x] Downstream: multilingual model on all four languages (Section 3)
- [ ] Resolve the balanced/curriculum data-provenance question in Section 5
- [ ] Decide how to disclose the multilingual model's training divergence (Section 4) in the
      paper — this is a real limitation on the multi-vs-mono comparison, not optional detail
- [x] Kannada checkpoint recoverability — confirmed unrecoverable (deleted from the SLURM
      cluster's local disk and never pushed to the Hub). No training curve is possible for
      Kannada; the single final data point in Section 4's local-seed-log table is all that exists.
