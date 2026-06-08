#!/usr/bin/env python3

"""

clean_merge.py

Cleans and merges CC100 + Wiki + Samanantar per language.

TinyStories is eval-only — excluded from merge.



Cleaning steps:

  1. Unicode NFC normalization

  2. Strip HTML/XML tags

  3. Strip URLs and emails



Output: $SCRATCH/dravidian/data/processed/{lang}_clean.txt



Usage:

    python clean_merge.py

    python clean_merge.py --lang te

"""



import re

import unicodedata

import argparse

from pathlib import Path

from tqdm import tqdm



BASE     = Path("/nfs/storage1/home/pulipakv/Dravidian")

RAW_DIR  = BASE / "data" / "raw"

PROC_DIR = BASE / "data" / "processed"

PROC_DIR.mkdir(parents=True, exist_ok=True)



LANGUAGES = ["te", "ta", "kn", "ml"]

SOURCES   = ["cc100", "wiki", "samanantar"]   # TinyStories excluded (eval only)



_RE_HTML = re.compile(r"<[^>]+>")

_RE_URL  = re.compile(r"https?://\S+|www\.\S+|\S+@\S+\.\S+")





def clean_line(line: str) -> str | None:

    line = unicodedata.normalize("NFC", line.strip())

    if not line:

        return None

    line = _RE_HTML.sub(" ", line)

    line = _RE_URL.sub(" ", line)

    line = " ".join(line.split())

    return line if line else None





def process_language(lang: str) -> None:

    out_path = PROC_DIR / f"{lang}_clean.txt"

    total_in = total_out = 0



    with open(out_path, "w", encoding="utf-8") as fout:

        for src in SOURCES:

            in_path = RAW_DIR / f"{lang}_{src}.txt"

            if not in_path.exists():

                print(f"  [{lang}] {in_path.name} not found — skipping")

                continue



            src_in = src_out = 0

            with open(in_path, encoding="utf-8", errors="ignore") as fin:

                for raw in tqdm(fin, desc=f"  {lang}/{src}", unit=" lines", leave=False):

                    src_in += 1

                    line = clean_line(raw)

                    if line:

                        fout.write(line + "\n")

                        src_out += 1



            total_in  += src_in

            total_out += src_out

            print(f"  [{lang}] {src:<12}  {src_in:>12,} in  →  {src_out:>10,} kept  ({100*src_out/max(src_in,1):.1f}%)")



    word_count = sum(len(l.split()) for l in open(out_path, encoding="utf-8"))

    print(f"  [{lang}] TOTAL        {total_in:>12,} in  →  {total_out:>10,} kept")

    print(f"  [{lang}] Words        {word_count:,}")

    print(f"  [{lang}] Output       {out_path}")





def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument("--lang", choices=LANGUAGES, default=None)

    args = parser.parse_args()

    langs = [args.lang] if args.lang else LANGUAGES



    print("=" * 62)

    print("  CLEANING + MERGING PIPELINE")

    print("=" * 62)



    for lang in langs:

        print(f"\n{'─'*62}\n  {lang.upper()}\n{'─'*62}")

        process_language(lang)



    print("\nDone.")





if __name__ == "__main__":

    main()

