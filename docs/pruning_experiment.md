# Data-Pruning Scaling-Law Experiment (Telugu)

## Research question

Does pruning 30-50% of low-resource pretraining data using a small reference
model preserve — or beat — the 100%-baseline validation BPB? If a 50% core-set
matches the full-data run, standard power-law scaling (more data → strictly
lower loss) doesn't hold once easy/redundant examples are pruned out.

This is motivated by two results:

- Ari Morcos' NeurIPS work: keeping *easy* examples helps when training data
  is scarce; keeping *hard* examples helps once data is abundant.
- Datology's use of small (125M-scale) reference models to score and prune
  pretraining data for downstream performance.

## Method

1. **Reference model**: the existing Telugu GPT-2 checkpoint,
   [`pulipakav-1/dravidian-gpt2-telugu`](https://huggingface.co/pulipakav-1/dravidian-gpt2-telugu)
   (the `seed2` run — eval_loss 3.7635 / ppl 43.10, see `results/raw/telugu_seed2.json`).
2. **Scoring**: pass every line of `data/splits/te/te_train.txt` (the existing
   96% train split, itself a merge of CC100 + Wikipedia + Samanantar) through
   the reference model and record its cross-entropy loss (nats/token).
   `python -m dravidian_lm.pruning.score`
3. **Pruned splits** (each ~50% of the train split), computed from the loss
   distribution: `python -m dravidian_lm.pruning.make_splits`
   - **Easy** (`D_easy`): bottom 50% by loss (lowest-loss half — "predictable"
     to the reference model).
   - **Hard** (`D_hard`): top 50% by loss (highest-loss half).
   - **Mid / Pareto core-set** (`D_mid`): the 40th-90th percentile band —
     cuts the bottom 40% ("trivial repetitions") and the top 10% ("noisy
     garbage"), keeping the middle 50%.
   - **Random** (`D_random`): uniform random 50% sample, seed 42 — the
     control that isolates the effect of *which* 50% is kept.

   Val/test splits (`te_val.txt` / `te_test.txt`) are never touched, so every
   variant and the 100% baseline are evaluated on identical held-out data.
4. **Training**: one 110M-parameter GPT-2 (the same architecture as the
   baseline — 12 layer/head, 768 dim, see `configs/models/gpt2-small.yaml`)
   trained from scratch on each 50% slice, under the *same* learning-rate
   schedule as the baseline (peak LR, warmup steps, cosine decay, weight
   decay, batch size — all unchanged; only the corpus differs). One seed per
   variant (seed 1). `python -m dravidian_lm.models.gpt2.train --variant
   {easy,hard,mid,random} --seed 1 --eval_strategy steps --eval_steps 500`

   The `--eval_strategy steps` flag makes each run log a full tokens-vs-BPB
   curve (`eval_history` in the result JSON), not just a single final number.
5. **Evaluation**: `python -m dravidian_lm.pruning.evaluate_variants` runs
   held-out BPB/perplexity (`evaluation.perplexity`) and WikiANN NER F1
   (`evaluation.downstream`) for every variant plus the 100% baseline
   (re-evaluated fresh from the HF Hub checkpoint, so the *final numbers* are
   directly comparable — only its *training curve* stays the coarse,
   already-recorded 3-point one, since it wasn't trained with step-level
   eval logging and isn't being re-trained just for this experiment).
6. **Deliverable**: `python -m dravidian_lm.analysis.plot_pruning_scaling`
   plots tokens trained vs validation BPB, comparing the 100% baseline
   against `D_mid` and `D_random` (plus `D_easy`/`D_hard` for the broader
   easy-vs-hard-at-scale story). `analysis.summarize_pruning` prints the
   companion BPB/perplexity/NER table.

## Reading the result

- If `D_mid` (or even `D_random`) reaches the baseline's final BPB using
  ~half the tokens, that's evidence pruning breaks the naive
  more-data-is-strictly-better scaling assumption for this corpus.
- Comparing `D_easy` vs `D_hard` against the Morcos framing: at this
  data scale (50% of an already low-resource corpus), we'd expect `D_easy`
  to do relatively better and `D_hard` relatively worse — the opposite would
  suggest the corpus isn't "scarce" in the way the framing assumes.
- WikiANN NER F1 is the downstream sanity check: a pruned variant that wins
  on BPB but loses on NER indicates the reference model's loss signal isn't
  fully aligned with downstream quality.

## Pipeline commands

```bash
export PYTHONPATH=src
export DRAVIDIAN_LM_BASE=$PWD

python -m dravidian_lm.pruning.score --language_code te
python -m dravidian_lm.pruning.make_splits --language_code te

for variant in easy hard mid random; do
  accelerate launch --num_processes 4 -m dravidian_lm.models.gpt2.train \
    --language telugu --tokenizer_name te --variant "$variant" \
    --seed 1 --eval_strategy steps --eval_steps 500
done

python -m dravidian_lm.pruning.evaluate_variants --language telugu --language_code te --tokenizer_name te
python -m dravidian_lm.analysis.plot_pruning_scaling --language telugu
python -m dravidian_lm.analysis.summarize_pruning --language telugu
```

Or run the whole thing as one SLURM job: [`scripts/run_pruning_experiment.sh`](../scripts/run_pruning_experiment.sh).

## Outputs

- `data/splits/te/te_train_scores.txt` — per-line reference-model loss.
- `data/splits/te/pruned/te_train_{easy,hard,mid,random}.txt` — pruned splits.
- `artifacts/models/gpt2/telugu/seed1_{easy,hard,mid,random}/` — checkpoints.
- `results/raw/telugu_seed1_{easy,hard,mid,random}.json` — training curves.
- `results/raw/telugu_pruning_{easy,hard,mid,random,full}.json` — combined eval.
- `results/plots/telugu_pruning_scaling_bpb.png` — the deliverable plot.
