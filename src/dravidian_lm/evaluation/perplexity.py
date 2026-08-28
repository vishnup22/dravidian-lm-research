from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, PreTrainedTokenizer

from dravidian_lm.paths import SPLITS_DIR


MAX_LENGTH = 1024
# Sources used during training; per-source test files are optional
SOURCES = ("cc100", "wiki", "samanantar")


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


@torch.no_grad()
def score_lines(
    texts: list[str],
    tokenizer: PreTrainedTokenizer,
    model: GPT2LMHeadModel,
    device: str,
    batch_size: int = 8,
) -> list[float]:
    """Compute per-line cross-entropy loss (nats/token) for use as a data-pruning score.

    Unlike `eval_perplexity`, which returns one loss averaged over an entire
    batch (the value HF's `outputs.loss` reports), this returns one loss per
    input line so lines can be ranked/pruned individually. Padding never
    contributes to a line's loss even though lines are batched together.
    """
    model.eval()
    scores: list[float] = []

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

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Shift so position t predicts token t+1, matching GPT-2's internal convention.
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous().float()

        token_nll = F.cross_entropy(
            shift_logits.transpose(1, 2), shift_labels, reduction="none"
        )
        token_nll = token_nll * shift_mask

        per_line_tokens = shift_mask.sum(dim=1).clamp(min=1.0)
        per_line_nll = token_nll.sum(dim=1) / per_line_tokens
        scores.extend(per_line_nll.cpu().tolist())

    return scores


def run_perplexity_suite(
    tokenizer: PreTrainedTokenizer,
    model: GPT2LMHeadModel,
    language_code: str,
    device: str,
    max_eval_lines: int = 5_000,
    batch_size: int = 8,
) -> dict:
    """Run perplexity on the overall test split and each per-source split."""
    results: dict = {}

    test_path = SPLITS_DIR / language_code / f"{language_code}_test.txt"
    if test_path.exists():
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
    else:
        print(f"  [perplexity] test split not found: {test_path}")

    per_source: dict = {}
    per_line_limit = max(1, max_eval_lines // len(SOURCES))
    for source in SOURCES:
        src_path = SPLITS_DIR / language_code / f"{language_code}_test_{source}.txt"
        if not src_path.exists():
            continue
        texts = load_texts(src_path, max_lines=per_line_limit)
        result = eval_perplexity(
            texts, tokenizer, model, device, source=source, batch_size=batch_size
        )
        per_source[source] = result.to_dict()
        print(
            f"  [perplexity] {source:<12}: "
            f"loss={result.eval_loss:.4f}  ppl={result.perplexity:.2f}  "
            f"bpb={result.bpb:.4f}"
        )

    if per_source:
        results["per_source"] = per_source
    else:
        print(
            "  [perplexity] per-source splits not found; "
            "create data/splits/te/te_test_{cc100,wiki,samanantar}.txt to enable"
        )

    return results
