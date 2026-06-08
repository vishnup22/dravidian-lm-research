#!/bin/bash
#SBATCH --job-name=dravidian
#SBATCH --partition=gpu-month-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=31-00:00:00
#SBATCH --output=logs/dravidian_%j.out
#SBATCH --error=logs/dravidian_%j.err

mkdir -p logs

source ~/.bashrc
conda activate telugu_llm

accelerate launch --num_processes 4 train_gpt2.py --language telugu --tokenizer_dirname te
accelerate launch --num_processes 4 train_gpt2.py --language tamil --tokenizer_dirname ta
accelerate launch --num_processes 4 train_gpt2.py --language kannada --tokenizer_dirname kn
accelerate launch --num_processes 4 train_gpt2.py --language malayalam --tokenizer_dirname ml