#!/usr/bin/env python3

from __future__ import annotations

import argparse

from dravidian_lm.paths import PROCESSED_DATA_DIR, SPLITS_DIR


LANGUAGES = ["te", "ta", "kn", "ml"]

TRAIN_RATIO = 0.96
VAL_RATIO = 0.02
TEST_RATIO = 0.02


def split_language(lang: str) -> None:
    in_path = PROCESSED_DATA_DIR / f"{lang}_clean.txt"
    if not in_path.exists():
        print(f"  [{lang}] {in_path.name} not found, skipping")
        return

    out_dir = SPLITS_DIR / lang
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  [{lang}] Counting lines...")
    with in_path.open(encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    train_end = int(total_lines * TRAIN_RATIO)
    val_end = train_end + int(total_lines * VAL_RATIO)

    train_path = out_dir / f"{lang}_train.txt"
    val_path = out_dir / f"{lang}_val.txt"
    test_path = out_dir / f"{lang}_test.txt"

    print(f"  [{lang}] Total lines : {total_lines:,}")
    print(f"  [{lang}] Train       : {train_end:,} ({TRAIN_RATIO * 100:.0f}%)")
    print(f"  [{lang}] Val         : {val_end - train_end:,} ({VAL_RATIO * 100:.0f}%)")
    print(f"  [{lang}] Test        : {total_lines - val_end:,} ({TEST_RATIO * 100:.0f}%)")

    with (
        in_path.open(encoding="utf-8") as fin,
        train_path.open("w", encoding="utf-8") as ftrain,
        val_path.open("w", encoding="utf-8") as fval,
        test_path.open("w", encoding="utf-8") as ftest,
    ):
        for idx, line in enumerate(fin):
            if idx < train_end:
                ftrain.write(line)
            elif idx < val_end:
                fval.write(line)
            else:
                ftest.write(line)

    print(f"  [{lang}] Done -> {train_path.name} / {val_path.name} / {test_path.name}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create train/val/test splits for processed corpora.")
    parser.add_argument("--lang", choices=LANGUAGES, default=None)
    args = parser.parse_args()
    langs = [args.lang] if args.lang else LANGUAGES

    print("=" * 62)
    print("  TRAIN / VAL / TEST SPLIT (96 / 2 / 2)")
    print("=" * 62 + "\n")

    for lang in langs:
        split_language(lang)

    print("All splits complete.")


if __name__ == "__main__":
    main()
