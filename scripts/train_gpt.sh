#!/bin/bash
#SBATCH --job-name=dravidian-gpt2
#SBATCH --partition=gpu-month-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=31-00:00:00
#SBATCH --output=logs/dravidian_gpt2_%j.out
#SBATCH --error=logs/dravidian_gpt2_%j.err

set -euo pipefail

mkdir -p logs

source ~/.bashrc
conda activate telugu_llm

export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export DRAVIDIAN_LM_BASE="${PWD}"

accelerate launch --num_processes 4 -m dravidian_lm.models.gpt2.train --language telugu --tokenizer_name te
accelerate launch --num_processes 4 -m dravidian_lm.models.gpt2.train --language tamil --tokenizer_name ta
accelerate launch --num_processes 4 -m dravidian_lm.models.gpt2.train --language kannada --tokenizer_name kn
accelerate launch --num_processes 4 -m dravidian_lm.models.gpt2.train --language malayalam --tokenizer_name ml
