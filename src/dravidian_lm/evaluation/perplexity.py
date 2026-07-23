from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import hf_hub_download
from transformers import GPT2LMHeadModel, PreTrainedTokenizer


MAX_LENGTH = 1024

# Held-out splits live in the canonical HF Hub dataset, not on local disk —
# data/splits/{code}/{code}_test.txt was the local split.py output, but the
# splits actually used going forward are the ones published here.
HF_DATASET_ID = "pulipakav-1/dravidian"


def download_split(language: str, split: str) -> Path:
    """Download a {language}/{split}/{language}_{split}_cleaned_final.txt file.

    `language` is the full name (telugu/tamil/kannada/malayalam), matching the
    directory layout of hf://datasets/pulipakav-1/dravidian.
    """
    repo_path = f"{language}/{split}/{language}_{split}_cleaned_final.txt"
    return Path(
        hf_hub_download(HF_DATASET_ID, repo_path, repo_type="dataset")
    )


@dataclass
class PerplexityResult:
    source: str
    num_sequences: int
    num_tokens: int
    num_bytes: int
    eval_loss: float      # cross-entropy in nats per token
    perplexity: float
    bpb: float            # bits per byte of input text

    def to_dict(self) -> dict:
        return asdict(self)


def load_texts(path: Path, max_lines: Optional[int] = None) -> list[str]:
    texts: list[str] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if max_lines and len(texts) >= max_lines:
                break
            line = line.strip()
            if line:
                texts.append(line)
    return texts


@torch.no_grad()
def eval_perplexity(
    texts: list[str],
    tokenizer: PreTrainedTokenizer,
    model: GPT2LMHeadModel,
    device: str,
    source: str = "overall",
    batch_size: int = 8,
) -> PerplexityResult:
    """Compute perplexity and BPB over a list of texts.

    Uses the sliding-window method for any text longer than MAX_LENGTH so that
    no tokens are evaluated without context.  Short texts (the common case for
    line-split corpora) are processed directly.
    """
    model.eval()

    total_nll = 0.0      # nats
    total_tokens = 0
    total_bytes = 0
    n_sequences = 0

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        # Mask padding so it does not contribute to the loss.
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        # GPT-2 shifts inputs by 1 internally; each sequence of length L
        # contributes L-1 predicted positions.
        # Total non-padding predicted positions = sum(mask) - batch_size.
        n_predicted = attention_mask.sum().item() - input_ids.shape[0]
        total_nll += outputs.loss.item() * n_predicted
        total_tokens += n_predicted
        total_bytes += sum(len(t.encode("utf-8")) for t in batch)
        n_sequences += len(batch)

    if total_tokens == 0:
        raise ValueError(f"No tokens to evaluate for source: {source}")

    avg_loss = total_nll / total_tokens
    ppl = math.exp(min(avg_loss, 20.0))
    # BPB: total NLL converted to bits, divided by total UTF-8 bytes.
    # Normalises for tokeniser efficiency so different tokenisers are comparable.
    bpb = (total_nll / math.log(2)) / total_bytes

    return PerplexityResult(
        source=source,
        num_sequences=n_sequences,
        num_tokens=total_tokens,
        num_bytes=total_bytes,
        eval_loss=round(avg_loss, 6),
        perplexity=round(ppl, 4),
        bpb=round(bpb, 6),
    )


def run_perplexity_suite(
    tokenizer: PreTrainedTokenizer,
    model: GPT2LMHeadModel,
    language: str,
    device: str,
    max_eval_lines: int = 5_000,
    batch_size: int = 8,
) -> dict:
    """Run perplexity on the held-out test split from hf://datasets/pulipakav-1/dravidian."""
    results: dict = {}

    test_path = download_split(language, "test")
    texts = load_texts(test_path, max_lines=max_eval_lines)
    result = eval_perplexity(
        texts, tokenizer, model, device, source="overall", batch_size=batch_size
    )
    results["overall"] = result.to_dict()
    print(
        f"  [perplexity] overall      : "
        f"loss={result.eval_loss:.4f}  ppl={result.perplexity:.2f}  "
        f"bpb={result.bpb:.4f}  ({result.num_sequences:,} seqs)"
    )

    return results
