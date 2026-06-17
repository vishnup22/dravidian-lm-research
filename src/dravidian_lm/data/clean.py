#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import unicodedata

from tqdm import tqdm

from dravidian_lm.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR


LANGUAGES = ["te", "ta", "kn", "ml"]
SOURCES = ["cc100", "wiki", "samanantar"]

RE_HTML = re.compile(r"<[^>]+>")
RE_URL = re.compile(r"https?://\S+|www\.\S+|\S+@\S+\.\S+")

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def clean_line(line: str) -> str | None:
    line = unicodedata.normalize("NFC", line.strip())
    if not line:
        return None
    line = RE_HTML.sub(" ", line)
    line = RE_URL.sub(" ", line)
    line = " ".join(line.split())
    return line if line else None


def process_language(lang: str) -> None:
    out_path = PROCESSED_DATA_DIR / f"{lang}_clean.txt"
    total_in = 0
    total_out = 0

    with out_path.open("w", encoding="utf-8") as fout:
        for src in SOURCES:
            in_path = RAW_DATA_DIR / f"{lang}_{src}.txt"
            if not in_path.exists():
                print(f"  [{lang}] {in_path.name} not found, skipping")
                continue

            src_in = 0
            src_out = 0
            with in_path.open(encoding="utf-8", errors="ignore") as fin:
                for raw in tqdm(fin, desc=f"{lang}/{src}", unit="lines", leave=False):
                    src_in += 1
                    line = clean_line(raw)
                    if line:
                        fout.write(line + "\n")
                        src_out += 1

            total_in += src_in
            total_out += src_out
            keep_rate = 100 * src_out / max(src_in, 1)
            print(f"  [{lang}] {src:<12} {src_in:>12,} in -> {src_out:>10,} kept ({keep_rate:.1f}%)")

    with out_path.open(encoding="utf-8") as cleaned:
        word_count = sum(len(line.split()) for line in cleaned)

    print(f"  [{lang}] TOTAL        {total_in:>12,} in -> {total_out:>10,} kept")
    print(f"  [{lang}] Words        {word_count:,}")
    print(f"  [{lang}] Output       {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and merge Dravidian text corpora.")
    parser.add_argument("--lang", choices=LANGUAGES, default=None)
    args = parser.parse_args()
    langs = [args.lang] if args.lang else LANGUAGES

    print("=" * 62)
    print("  CLEANING + MERGING PIPELINE")
    print("=" * 62)

    for lang in langs:
        print(f"\n{'-' * 62}\n  {lang.upper()}\n{'-' * 62}")
        process_language(lang)

    print("\nDone.")


if __name__ == "__main__":
    main()
