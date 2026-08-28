#!/usr/bin/env python3

from __future__ import annotations

"""Build easy/hard/mid/random pruned training splits from reference-model scores.

Step 2 of the data-pruning scaling-law experiment. Consumes the per-line
scores produced by `score.py` and partitions the corresponding train split
into four 50% subsets:

  easy   — bottom 50% by loss (lowest-loss half)
  hard   — top 50% by loss (highest-loss half)
  mid    — 40th-90th percentile band (cuts the bottom 40% "trivial" lines
           and the top 10% "noisy garbage" lines)
  random — uniform random 50% sample (seeded control)

Usage
-----
python -m dravidian_lm.pruning.make_splits --language_code te
"""

import argparse

import numpy as np

from dravidian_lm.paths import SPLITS_DIR


VARIANTS = ("easy", "hard", "mid", "random")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Partition a scored train split into easy/hard/mid/random pruned subsets."
    )
    parser.add_argument("--language_code", required=True, help="e.g. te")
    parser.add_argument(
        "--split",
        default="train",
        help="Base split name matching the one passed to score.py (default: train).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for the random-control split.")
    return parser.parse_args()


def build_masks(scores: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    valid = ~np.isnan(scores)
    valid_scores = scores[valid]
    if valid_scores.size == 0:
        raise ValueError("No valid (non-nan) scores to split on.")

    median = np.median(valid_scores)
    p40, p90 = np.percentile(valid_scores, [40, 90])

    easy_mask = valid & (scores <= median)
    hard_mask = valid & (scores > median)
    mid_mask = valid & (scores >= p40) & (scores < p90)

    rng = np.random.default_rng(seed)
    valid_indices = np.nonzero(valid)[0]
    n_random = len(valid_indices) // 2
    chosen = rng.choice(valid_indices, size=n_random, replace=False)
    random_mask = np.zeros(scores.shape[0], dtype=bool)
    random_mask[chosen] = True

    print(f"  Valid lines   : {valid.sum():,} / {scores.shape[0]:,}")
    print(f"  median loss   : {median:.4f}")
    print(f"  p40 / p90 loss: {p40:.4f} / {p90:.4f}")
    for name, mask in (
        ("easy", easy_mask),
        ("hard", hard_mask),
        ("mid", mid_mask),
        ("random", random_mask),
    ):
        pct = 100 * mask.sum() / valid.sum()
        print(f"  {name:<7}: {mask.sum():>12,} lines ({pct:.1f}% of valid)")

    return {"easy": easy_mask, "hard": hard_mask, "mid": mid_mask, "random": random_mask}


def main() -> None:
    args = parse_args()
    lang = args.language_code

    scores_path = SPLITS_DIR / lang / f"{lang}_{args.split}_scores.txt"
    train_path = SPLITS_DIR / lang / f"{lang}_{args.split}.txt"
    if not scores_path.exists():
        raise FileNotFoundError(f"Scores file not found: {scores_path} (run pruning.score first)")
    if not train_path.exists():
        raise FileNotFoundError(f"Train split not found: {train_path}")

    print(f"[{lang}] Loading scores from {scores_path} ...")
    scores = np.loadtxt(scores_path, dtype=np.float32)

    print(f"[{lang}] Computing split masks (seed={args.seed}) ...")
    masks = build_masks(scores, args.seed)

    out_dir = SPLITS_DIR / lang / "pruned"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_paths = {v: out_dir / f"{lang}_{args.split}_{v}.txt" for v in VARIANTS}
    out_handles = {v: p.open("w", encoding="utf-8") for v, p in out_paths.items()}

    print(f"[{lang}] Writing pruned splits to {out_dir} ...")
    try:
        with train_path.open(encoding="utf-8", errors="replace") as fin:
            for idx, line in enumerate(fin):
                if idx >= scores.shape[0]:
                    break
                for variant in VARIANTS:
                    if masks[variant][idx]:
                        out_handles[variant].write(line)
    finally:
        for handle in out_handles.values():
            handle.close()

    for variant, path in out_paths.items():
        print(f"  [{lang}] {variant:<7} -> {path}")
    print("Done.")


if __name__ == "__main__":
    main()
