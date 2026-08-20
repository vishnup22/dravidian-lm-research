#!/usr/bin/env python3

from __future__ import annotations

"""Score a training corpus, line by line, with a small reference model.

This is step 1 of the data-pruning scaling-law experiment: pass the
uncurated train split through an already-trained checkpoint and record each
line's cross-entropy loss (nats/token). `make_splits.py` then uses these
scores to build easy/hard/mid/random pruned training sets.

Usage
-----
python -m dravidian_lm.pruning.score --language_code te
"""

import argparse
import time

from tqdm import tqdm

from dravidian_lm.paths import SPLITS_DIR
from dravidian_lm.evaluation.perplexity import score_lines
from dravidian_lm.evaluation.run_eval import MODEL_ID, load_model_and_tokenizer


READ_CHUNK_LINES = 2048


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a train split with a reference model for data pruning."
    )
    parser.add_argument("--language_code", required=True, help="e.g. te")
    parser.add_argument(
        "--split",
        default="train",
        help="Which split file to score (default: train, the one make_splits.py prunes).",
    )
    parser.add_argument(
        "--model_name",
        default=MODEL_ID,
        help="HF Hub id or local path of the reference scoring model.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help=(
            "Inference-only (no backward pass/optimizer state), so this can be much "
            "larger than a training batch size -- raise further if GPU memory allows."
        ),
    )
    parser.add_argument("--device", default=None, help="Device override, auto-detects if omitted.")
    parser.add_argument(
        "--max_lines",
        type=int,
        default=None,
        help="Optional cap for dry runs; omit to score the whole corpus.",
    )
    return parser.parse_args()


def iter_line_chunks(path, chunk_size: int, max_lines: int | None):
    chunk: list[str] = []
    total = 0
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            chunk.append(line)  # keep blanks so line numbers stay aligned with the source file
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
            total += 1
            if max_lines and total >= max_lines:
                break
    if chunk:
        yield chunk


def main() -> None:
    args = parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    in_path = SPLITS_DIR / args.language_code / f"{args.language_code}_{args.split}.txt"
    out_path = SPLITS_DIR / args.language_code / f"{args.language_code}_{args.split}_scores.txt"
    if not in_path.exists():
        raise FileNotFoundError(f"Split not found: {in_path}")

    print(f"Scoring {in_path} with {args.model_name} on {device}")
    model, tokenizer = load_model_and_tokenizer(args.model_name, device)

    start = time.time()
    n_scored = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for chunk in tqdm(
            iter_line_chunks(in_path, READ_CHUNK_LINES, args.max_lines),
            desc=f"{args.language_code}/{args.split} scoring",
            unit="chunk",
        ):
            # Empty lines can't be scored (no tokens); write a sentinel so line
            # numbers in the scores file stay aligned with in_path.
            non_empty = [line for line in chunk if line]
            if non_empty:
                scores = score_lines(non_empty, tokenizer, model, device, batch_size=args.batch_size)
            else:
                scores = []
            score_iter = iter(scores)
            for line in chunk:
                if line:
                    fout.write(f"{next(score_iter):.6f}\n")
                else:
                    fout.write("nan\n")
                n_scored += 1

    elapsed = time.time() - start
    print(f"Scored {n_scored:,} lines in {elapsed:.0f}s -> {out_path}")


if __name__ == "__main__":
    main()
