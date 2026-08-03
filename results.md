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

## 2. Tokenizer efficiency **(unverified — Telugu only, not yet saved to disk)**

Source: console output from an interrupted run; never written to `results/raw/`. Tamil, Kannada,
Malayalam not yet run. Sample: 2,000 test-split sentences.

| Tokenizer | Vocab | Fertility (tok/word) | Compression (bytes/tok) | UNK rate |
|---|---:|---:|---:|---:|
| dravidian-gpt2-telugu (ours) | 32,000 | 1.61 | 13.12 | 0.00% |
| XLM-R | 250,002 | 2.37 | 8.93 | 0.00% |
| mBERT | 119,547 | 3.88 | 5.45 | 0.24% |
| mGPT | 100,000 | 6.19 | 3.41 | 0.00% |

Our dedicated tokenizer is markedly more efficient than every multilingual baseline despite a
much smaller vocabulary (32K vs. 100K–250K) — lower fertility and higher compression than
XLM-R/mBERT/mGPT, and zero UNKs. **TODO: re-run for Tamil, Kannada, Malayalam and persist to
`results/raw/`.**

## 3. Downstream fine-tuning (verified against console output, pasted twice consistently)

Full fine-tuning, 5 epochs, IndicSentiment (accuracy) + WikiANN NER (span F1), vs. mGPT baseline
fine-tuned identically. IndicSentiment test set is small (~24 examples, self-split from
IndicSentiment's `validation` data — see Section 5 caveat); NER test set is the standard WikiANN
1,000-example split.

| Language | Model | IndicSentiment acc. | mGPT acc. | NER F1 | mGPT F1 |
|---|---|---:|---:|---:|---:|
| Telugu | Monolingual | **0.7083** | 0.6250 | **0.6092** | 0.3039 |
| Telugu | Multilingual | 0.6667 | 0.5417 | 0.5605 | 0.2699 |
| Tamil | Monolingual | 0.4167 | **0.6250** | 0.2031 | **0.2809** |
| Kannada | Monolingual | 0.5833 | **0.6250** | 0.1676 | **0.2845** |
| Malayalam | Monolingual | **0.5417** | 0.5000 | 0.2054 | **0.2773** |

IndicXNLI was attempted but is permanently out of scope: `ai4bharat/IndicXNLI` uses a deprecated
Hub loading script, and none of our four languages are covered by the `facebook/xnli` fallback.

Notes:
- Telugu (both mono and multi) clearly beats mGPT on both tasks. Tamil and Kannada lose to
  mGPT on both tasks. Malayalam is roughly split (slightly ahead on sentiment, behind on NER).
  This asymmetry is worth investigating rather than reporting flat — it doesn't obviously
  track corpus size or perplexity rank (Malayalam has the *best* perplexity of the four but a
  middling downstream result).
- **IndicSentiment accuracy is noisy at n≈24 test examples** — one example = ~4 points of
  accuracy. Don't treat small deltas (e.g. Malayalam's 0.5417 vs 0.5000) as meaningful; NER F1
  at n=1,000 is the more trustworthy of the two downstream numbers.
- Multi-model downstream scores only exist for Telugu so far; Tamil/Kannada/Malayalam not yet
  run for the multilingual model.

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
and `results/figures/{training_loss,validation_perplexity}.{png,pdf}`). Covers Telugu, Tamil,
Malayalam, and the multilingual model — **Kannada has no checkpoint subfolders uploaded to the
Hub**, only the final merged model, so no curve exists for it beyond the single point above.

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

- [ ] Tokenizer efficiency: Tamil, Kannada, Malayalam (Telugu done but unsaved — re-run all four)
- [ ] Downstream: confirm the four monolingual `_full_eval.json` files are pushed/pulled and
      correctly attributed (the Telugu file was previously mislabeled — verify it now says
      `dravidian-gpt2-telugu`, not `-multi`)
- [ ] Downstream: multilingual model on Tamil, Kannada, Malayalam (only Telugu done)
- [ ] Resolve the balanced/curriculum data-provenance question in Section 5
- [ ] Decide how to disclose the multilingual model's training divergence (Section 4) in the
      paper — this is a real limitation on the multi-vs-mono comparison, not optional detail
- [ ] Kannada has no checkpoint history on the Hub — confirm whether it's recoverable from the
      SLURM cluster's local disk if a full curve is wanted for consistency with the other three
