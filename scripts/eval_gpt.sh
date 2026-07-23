#!/bin/bash
#SBATCH --job-name=dravidian-gpt2-eval
#SBATCH --partition=gpu-month-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/dravidian_gpt2_eval_%j.out
#SBATCH --error=logs/dravidian_gpt2_eval_%j.err

set -euo pipefail

mkdir -p logs

source ~/.bashrc
conda activate telugu_llm

export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export DRAVIDIAN_LM_BASE="${PWD}"

# Perplexity + tokenizer-efficiency are language-generic; downstream now
# supports ta/kn/ml in addition to te (see downstream.py language_code plumbing).
# --run_baselines re-runs each downstream task with mGPT for a fair comparison.

python -m dravidian_lm.evaluation.run_eval \
    --model_name pulipakav-1/dravidian-gpt2-telugu \
    --language telugu --language_code te \
    --tasks perplexity tokenizer downstream --run_baselines

python -m dravidian_lm.evaluation.run_eval \
    --model_name pulipakav-1/dravidian-gpt2-tamil \
    --language tamil --language_code ta \
    --tasks perplexity tokenizer downstream --run_baselines

python -m dravidian_lm.evaluation.run_eval \
    --model_name pulipakav-1/dravidian-gpt2-kannada \
    --language kannada --language_code kn \
    --tasks perplexity tokenizer downstream --run_baselines

python -m dravidian_lm.evaluation.run_eval \
    --model_name pulipakav-1/dravidian-gpt2-malayalam \
    --language malayalam --language_code ml \
    --tasks perplexity tokenizer downstream --run_baselines
