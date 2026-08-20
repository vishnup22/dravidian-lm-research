#!/usr/bin/env python3

from __future__ import annotations

"""Print a markdown comparison table across the pruning-experiment variants.

Reads results/raw/{language}_pruning_*.json (written by
pruning.evaluate_variants) and reports eval_loss / perplexity / BPB /
WikiANN NER F1 / approximate train tokens for each variant, in one table.

Usage
-----
python -m dravidian_lm.analysis.summarize_pruning --language telugu
"""

import argparse
import json

from dravidian_lm.paths import RAW_RESULTS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize data-pruning experiment results.")
    parser.add_argument("--language", default="telugu")
    return parser.parse_args()


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def main() -> None:
    args = parse_args()
    paths = sorted(RAW_RESULTS_DIR.glob(f"{args.language}_pruning_*.json"))
    if not paths:
        print(f"No pruning results found for {args.language} in {RAW_RESULTS_DIR}")
        return

    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as f:
            payload = json.load(f)

        overall = payload.get("perplexity", {}).get("overall", {})
        ner = payload.get("wikiann_ner", {})
        history = payload.get("eval_history") or []
        final_tokens = history[-1]["tokens_seen"] if history else None

        rows.append(
            {
                "variant": payload.get("variant", path.stem),
                "eval_loss": overall.get("eval_loss"),
                "perplexity": overall.get("perplexity"),
                "bpb": overall.get("bpb"),
                "ner_f1": ner.get("score") if isinstance(ner, dict) else None,
                "tokens": final_tokens,
            }
        )

    print(f"# Pruning experiment: {args.language}\n")
    print("| Variant | Eval Loss | Perplexity | BPB | WikiANN NER F1 | Train Tokens |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        tokens_str = f"{row['tokens']:,}" if row["tokens"] else "-"
        print(
            f"| {row['variant']} | {fmt(row['eval_loss'])} | {fmt(row['perplexity'], 2)} | "
            f"{fmt(row['bpb'], 4)} | {fmt(row['ner_f1'])} | {tokens_str} |"
        )


if __name__ == "__main__":
    main()
