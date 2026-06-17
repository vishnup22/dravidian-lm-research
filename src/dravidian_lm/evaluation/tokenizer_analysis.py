from __future__ import annotations

from dataclasses import asdict, dataclass

from transformers import AutoTokenizer, PreTrainedTokenizer


# Multilingual tokenisers used as baselines in tokeniser-efficiency analysis.
# Fertility and compression ratios contextualise how our monolingual BPE
# compares against tokenisers that were never optimised for Telugu.
MULTILINGUAL_BASELINES: dict[str, str] = {
    "xlm-roberta-base": "xlm-roberta-base",
    "mbert": "bert-base-multilingual-cased",
    "mgpt": "ai-forever/mGPT",
}

SAMPLE_SIZE = 2_000   # number of sentences used for analysis


@dataclass
class TokenizerStats:
    name: str
    vocab_size: int
    fertility: float          # avg tokens per whitespace-split word (lower → more efficient)
    compression_ratio: float  # avg UTF-8 bytes per token (higher → denser representation)
    unk_rate: float           # fraction of output tokens that are UNK (lower → better coverage)

    def to_dict(self) -> dict:
        return asdict(self)


def analyze_tokenizer(
    texts: list[str],
    tokenizer: PreTrainedTokenizer,
    name: str,
) -> TokenizerStats:
    total_words = 0
    total_tokens = 0
    total_bytes = 0
    total_unk = 0
    unk_id = tokenizer.unk_token_id

    for text in texts:
        words = text.split()
        if not words:
            continue
        total_words += len(words)
        total_bytes += len(text.encode("utf-8"))
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        total_tokens += len(ids)
        if unk_id is not None:
            total_unk += sum(1 for t in ids if t == unk_id)

    return TokenizerStats(
        name=name,
        vocab_size=tokenizer.vocab_size,
        fertility=round(total_tokens / total_words, 4) if total_words else 0.0,
        compression_ratio=round(total_bytes / total_tokens, 4) if total_tokens else 0.0,
        unk_rate=round(total_unk / total_tokens, 6) if total_tokens else 0.0,
    )


def run_tokenizer_comparison(
    texts: list[str],
    our_tokenizer: PreTrainedTokenizer,
    our_name: str = "dravidian-gpt2-telugu",
) -> dict:
    """Compare our custom Telugu tokeniser against multilingual baselines.

    Metrics reported:
      fertility       — avg tokens per word; lower = more efficient segmentation
      compression_ratio — avg UTF-8 bytes per token; higher = denser per-token info
      unk_rate        — fraction of UNK tokens; lower = better script coverage
    """
    sample = texts[:SAMPLE_SIZE]
    results: dict = {}

    stats = analyze_tokenizer(sample, our_tokenizer, name=our_name)
    results[our_name] = stats.to_dict()
    _print_row(stats)

    for display_name, hf_id in MULTILINGUAL_BASELINES.items():
        try:
            tok = AutoTokenizer.from_pretrained(hf_id)
            stats = analyze_tokenizer(sample, tok, name=display_name)
            results[display_name] = stats.to_dict()
            _print_row(stats)
        except Exception as exc:
            print(f"  [tokenizer] {display_name:<30}: skipped — {exc}")

    return results


def _print_row(stats: TokenizerStats) -> None:
    print(
        f"  [tokenizer] {stats.name:<30}  "
        f"vocab={stats.vocab_size:>6,}  "
        f"fertility={stats.fertility:.2f} tok/word  "
        f"compression={stats.compression_ratio:.2f} bytes/tok  "
        f"unk={stats.unk_rate:.4%}"
    )
