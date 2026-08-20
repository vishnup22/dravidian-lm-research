#!/bin/bash
#SBATCH --job-name=dravidian-pruning
#SBATCH --partition=gpu-month-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=14-00:00:00
#SBATCH --output=logs/dravidian_pruning_%j.out
#SBATCH --error=logs/dravidian_pruning_%j.err

# Data-pruning scaling-law experiment (Telugu). See docs/pruning_experiment.md.
#
# Trains 4 GPT-2s from scratch on 50% pruned slices of the existing Telugu
# train split (easy / hard / mid / random) and compares them, plus the
# existing 100% baseline (telugu_seed2), on held-out BPB and WikiANN NER.
#
# Each variant is trained on a single GPU here; scale --gres/--nodes and add
# `accelerate launch --num_processes N` (as scripts/train_gpt.sh does) if you
# want multi-GPU per run.

set -euo pipefail

mkdir -p logs

source ~/.bashrc
conda activate telugu_llm

export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export DRAVIDIAN_LM_BASE="${PWD}"

LANG_CODE=te
LANGUAGE=telugu
TOKENIZER=te
SEED=1

echo "=== 1/5: scoring train split with the reference model ==="
python -m dravidian_lm.pruning.score --language_code "${LANG_CODE}"

echo "=== 2/5: building easy/hard/mid/random pruned splits ==="
python -m dravidian_lm.pruning.make_splits --language_code "${LANG_CODE}"

echo "=== 3/5: training each variant (seed=${SEED}) ==="
for variant in easy hard mid random; do
  python -m dravidian_lm.models.gpt2.train \
    --language "${LANGUAGE}" \
    --tokenizer_name "${TOKENIZER}" \
    --variant "${variant}" \
    --seed "${SEED}" \
    --eval_strategy steps \
    --eval_steps 500
done

echo "=== 4/5: evaluating all variants + the 100% baseline ==="
python -m dravidian_lm.pruning.evaluate_variants \
  --language "${LANGUAGE}" \
  --language_code "${LANG_CODE}" \
  --tokenizer_name "${TOKENIZER}" \
  --seed "${SEED}"

echo "=== 5/5: plot + summary table ==="
python -m dravidian_lm.analysis.plot_pruning_scaling --language "${LANGUAGE}"
python -m dravidian_lm.analysis.summarize_pruning --language "${LANGUAGE}"
