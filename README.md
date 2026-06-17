# Dravidian LM Research

Research repository for Dravidian language model pretraining, tokenization, and benchmarking across architectures.

This repo is organized as a research artifact rather than a one-off training folder. It is intended to support paper-grade experiments on Dravidian languages with a reproducible pipeline for data preparation, tokenizer training, model training, and result tracking.

## Topics

`dravidian-languages`, `language-modeling`, `pretraining`, `nlp`, `transformers`, `huggingface`, `gpt2`, `tokenization`, `low-resource-languages`, `computational-linguistics`

## Research Scope

Current focus:

- monolingual language modeling for Telugu, Kannada, Tamil, and Malayalam
- corpus collection from CC100, Wikipedia, Samanantar, and TinyStories
- SentencePiece tokenizer training
- GPT-2 pretraining as the first implemented architecture
- result logging for cross-language and cross-run comparison

Planned direction:

- additional architectures beyond GPT-2
- broader evaluation and benchmarking
- config-driven experiment execution

## Repository Layout

```text
.
|-- README.md
|-- requirements.txt
|-- configs/
|-- docs/
|-- notebooks/
|-- results/
|   `-- raw/
|-- scripts/
`-- src/
    `-- dravidian_lm/
        |-- analysis/
        |-- data/
        |-- models/
        `-- tokenization/
```

## Key Paths

- [src/dravidian_lm/data](./src/dravidian_lm/data): corpus download, cleaning, and splitting
- [src/dravidian_lm/tokenization](./src/dravidian_lm/tokenization): tokenizer training
- [src/dravidian_lm/models/gpt2](./src/dravidian_lm/models/gpt2): GPT-2 training
- [src/dravidian_lm/analysis](./src/dravidian_lm/analysis): result summarization
- [scripts/train_gpt.sh](./scripts/train_gpt.sh): cluster launcher
- [docs/reproducibility.md](./docs/reproducibility.md): execution and layout notes
- [configs](./configs): experiment templates for future config-driven runs

## Current Models

- Telugu: [pulipakav-1/dravidian-gpt2-telugu](https://huggingface.co/pulipakav-1/dravidian-gpt2-telugu) using the `seed2` subfolder
- Kannada: [pulipakav-1/dravidian-gpt2-kannada](https://huggingface.co/pulipakav-1/dravidian-gpt2-kannada)
- Malayalam: training in progress

## Current Results

Tracked raw outputs:

- [results/raw/telugu_seed2.json](./results/raw/telugu_seed2.json)
- [results/raw/kannada_seed1.json](./results/raw/kannada_seed1.json)

Sample metrics:

| Language | Run | Eval Loss | Perplexity |
| --- | --- | ---: | ---: |
| Telugu | seed2 | 3.7635 | 43.10 |
| Kannada | seed1 | 3.9794 | 53.49 |

## Reproducibility

Set up the environment:

```bash
pip install -r requirements.txt
export PYTHONPATH=src
export DRAVIDIAN_LM_BASE=$PWD
```

Run the pipeline:

```bash
python -m dravidian_lm.data.download --lang te
python -m dravidian_lm.data.clean --lang te
python -m dravidian_lm.data.split --lang te
python -m dravidian_lm.tokenization.train_tokenizer --lang te
python -m dravidian_lm.models.gpt2.train --language telugu --tokenizer_name te
python -m dravidian_lm.analysis.summarize_results
```

More detail is in [docs/reproducibility.md](./docs/reproducibility.md).

## Notes

- large corpora, tokenizers, checkpoints, and split artifacts are intentionally git-ignored
- experiment outputs are separated from source code
- notebooks are kept outside the core pipeline
- config templates are present now; wiring the runners directly to YAML is the next step
