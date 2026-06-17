# Dravidian LM Research

Research codebase for pretraining and evaluating language models for Dravidian languages, with an initial focus on monolingual GPT-2 models for Telugu, Kannada, Tamil, and Malayalam.

## About

Research repository for Dravidian language model pretraining, tokenization, and benchmarking across architectures.

## Topics

`dravidian-languages`, `language-modeling`, `pretraining`, `nlp`, `transformers`, `huggingface`, `gpt2`, `tokenization`, `low-resource-languages`, `computational-linguistics`

## Scope

This repository is intended for research experiments on:

- corpus collection and cleaning for Dravidian languages
- train/validation/test dataset preparation
- monolingual and joint tokenizer training
- causal language model pretraining
- cross-model and cross-language comparison for a paper workflow

The current pipeline is designed so additional architectures can be added without renaming or restructuring the project.

## Languages

- Telugu
- Kannada
- Tamil
- Malayalam

## Current Models

- Telugu: [pulipakav-1/dravidian-gpt2-telugu](https://huggingface.co/pulipakav-1/dravidian-gpt2-telugu)
- Kannada: [pulipakav-1/dravidian-gpt2-kannada](https://huggingface.co/pulipakav-1/dravidian-gpt2-kannada)
- Malayalam: training in progress

## Repository Structure

- [download_data.py](./download_data.py): download raw corpora from CC100, Wikipedia, Samanantar, and TinyStories
- [clean.py](./clean.py): clean and merge raw text sources
- [split.py](./split.py): create train/validation/test splits
- [tokenizer_utils.py](./tokenizer_utils.py): train monolingual and joint SentencePiece tokenizers
- [train_gpt2.py](./train_gpt2.py): pretrain GPT-2 language models and save evaluation logs
- [hindi.sh](./hindi.sh): multi-GPU cluster launch script for training runs
- [results](./results): saved experiment metrics

## Data Pipeline

1. Download corpora for each language.
2. Clean and merge the raw text.
3. Split the processed data into train, validation, and test sets.
4. Train language-specific or joint tokenizers.
5. Pretrain language models.
6. Record evaluation loss and perplexity for comparison across runs.

## Training Setup

The current training code uses:

- Hugging Face `transformers`
- Hugging Face `datasets`
- SentencePiece tokenization
- multi-GPU launch with `accelerate`

The initial implemented architecture is GPT-2, but the repository is meant to support additional architectures for the research paper.

## Current Results

Available result files:

- [results/telugu_seed1.json](./results/telugu_seed1.json)
- [results/telugu_seed2.json](./results/telugu_seed2.json)
- [results/kannada_seed1.json](./results/kannada_seed1.json)

Sample reported metrics from current runs:

| Language | Run | Eval Loss | Perplexity |
| --- | --- | ---: | ---: |
| Telugu | seed1 | 3.7620 | 43.03 |
| Telugu | seed2 | 3.7635 | 43.10 |
| Kannada | seed1 | 3.9794 | 53.49 |

## Notes

- This repository currently contains research code and experiment outputs.
- More architectures, baselines, and language coverage are expected as the paper develops.
- Malayalam training is currently in progress.
