#!/bin/bash
#SBATCH --job-name=d_multi
#SBATCH --partition=gpu-month-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=31-00:00:00
#SBATCH --output=logs/multi_%j.out
#SBATCH --error=logs/multi_%j.err

mkdir -p logs

source ~/.bashrc
conda activate telugu_llm

accelerate launch --num_processes 4 train_multilingual.py \
    --languages te,ta,kn,ml \
    --tokenizer_dirname joint
