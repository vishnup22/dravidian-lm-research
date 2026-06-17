# Reproducibility Notes

## Environment

- install dependencies from `requirements.txt`
- set `PYTHONPATH=src`
- optionally set `DRAVIDIAN_LM_BASE` if running outside the repo root

## Directory assumptions

The code now uses a project-relative layout:

- `data/raw/`
- `data/processed/`
- `data/splits/`
- `artifacts/tokenizers/`
- `artifacts/models/`
- `results/raw/`

## Typical workflow

1. Download raw corpora.
2. Clean and merge each language corpus.
3. Create train/val/test splits.
4. Train tokenizers.
5. Train models.
6. Summarize result JSON files.

## Example commands

```bash
export PYTHONPATH=src
export DRAVIDIAN_LM_BASE=$PWD

python -m dravidian_lm.data.download --lang te
python -m dravidian_lm.data.clean --lang te
python -m dravidian_lm.data.split --lang te
python -m dravidian_lm.tokenization.train_tokenizer --lang te
python -m dravidian_lm.models.gpt2.train --language telugu --tokenizer_name te
python -m dravidian_lm.analysis.summarize_results
```
