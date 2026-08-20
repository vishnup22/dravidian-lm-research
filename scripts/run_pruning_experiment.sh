#!/bin/bash
#SBATCH --job-name=dravidian-pruning
#SBATCH --partition=gpu-month-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/dravidian_pruning_%j.out
#SBATCH --error=logs/dravidian_pruning_%j.err

# Data-pruning scaling-law experiment (Telugu only). See docs/pruning_experiment.md.
#
# Trains 4 GPT-2s from scratch on 50% pruned slices of the existing Telugu
# train split (easy / hard / mid / random) and compares them, plus the
# existing 100% baseline (telugu_seed2), on held-out BPB and WikiANN NER.
#
# Each variant trains on 4 GPUs via accelerate launch, matching train_gpt.sh
# and the telugu_seed2 baseline's config. Baseline reference: 3 epochs /
# 268k steps / 4 GPUs took 67.1h (results/raw/telugu_seed2.json). Each
# pruned variant here is ~50% of the data on the same 4-GPU config, so
# expect roughly ~1.4 days/variant, ~5.6 days for all 4 sequentially, plus
# scoring + eval on top -- against the cluster's 7-day cap that's tight,
# with little margin.
#
# If it times out mid-run: just `sbatch` this same script again. It's
# resumable at two levels -- train_one() skips any variant whose result
# JSON already exists, and (as of this fix) resumes a partially-trained
# variant from its latest checkpoint instead of restarting it from scratch.

set -eo pipefail  # not -u: conda's activation hook references unset vars internally

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
python -m dravidian_lm.pruning.score --language_code "${LANG_CODE}" --batch_size 64

echo "=== 2/5: building easy/hard/mid/random pruned splits ==="
python -m dravidian_lm.pruning.make_splits --language_code "${LANG_CODE}"

echo "=== 3/5: training each variant (seed=${SEED}, 4 GPUs) ==="
for variant in easy hard mid random; do
  accelerate launch --num_processes 4 -m dravidian_lm.models.gpt2.train \
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
