#!/usr/bin/env python3

from __future__ import annotations

import argparse

from datasets import load_dataset
from tqdm import tqdm

from dravidian_lm.paths import RAW_DATA_DIR


LANGUAGES = ["te", "ta", "kn", "ml"]

CC100_LANG = {
    "te": "te",
    "ta": "ta",
    "kn": "kn",
    "ml": "ml",
}

WIKI_LANG = {
    "te": "20231101.te",
    "ta": "20231101.ta",
    "kn": "20231101.kn",
    "ml": "20231101.ml",
}

SAMANANTAR_LANG = {
    "te": "te",
    "ta": "ta",
    "kn": "kn",
    "ml": "ml",
}

TINYSTORIES_SPLITS = {
    "te": "te_cleaned.jsonl",
    "ta": "ta_cleaned.jsonl",
    "kn": "kn_cleaned.jsonl",
    "ml": "ml_cleaned.jsonl",
}

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_cc100(lang: str) -> None:
    out = RAW_DATA_DIR / f"{lang}_cc100.txt"
    if out.exists():
        print(f"  [{lang}] cc100 already exists, skipping")
        return
    print(f"  [{lang}] Downloading CC100...")
    ds = load_dataset("cc100", CC100_LANG[lang], split="train", streaming=True, trust_remote_code=True)
    written = 0
    with out.open("w", encoding="utf-8") as f:
        for row in tqdm(ds, desc=f"{lang}/cc100", unit="lines"):
            text = row.get("text", "").strip()
            if text:
                f.write(text + "\n")
                written += 1
    print(f"  [{lang}] cc100 done: {written:,} lines -> {out.name}")


def download_wiki(lang: str) -> None:
    out = RAW_DATA_DIR / f"{lang}_wiki.txt"
    if out.exists():
        print(f"  [{lang}] wiki already exists, skipping")
        return
    print(f"  [{lang}] Downloading Wikipedia...")
    ds = load_dataset(
        "wikimedia/wikipedia",
        WIKI_LANG[lang],
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    written = 0
    with out.open("w", encoding="utf-8") as f:
        for row in tqdm(ds, desc=f"{lang}/wiki", unit="articles"):
            text = row.get("text", "").strip()
            if not text:
                continue
            for line in text.splitlines():
                line = line.strip()
                if line:
                    f.write(line + "\n")
                    written += 1
    print(f"  [{lang}] wiki done: {written:,} lines -> {out.name}")


def download_samanantar(lang: str) -> None:
    out = RAW_DATA_DIR / f"{lang}_samanantar.txt"
    if out.exists():
        print(f"  [{lang}] samanantar already exists, skipping")
        return
    print(f"  [{lang}] Downloading Samanantar...")
    ds = load_dataset(
        "ai4bharat/samanantar",
        SAMANANTAR_LANG[lang],
        split="train",
        streaming=True,
    )
    written = 0
    with out.open("w", encoding="utf-8") as f:
        for row in tqdm(ds, desc=f"{lang}/samanantar", unit="lines"):
            text = row.get("tgt", "").strip()
            if text:
                f.write(text + "\n")
                written += 1
    print(f"  [{lang}] samanantar done: {written:,} lines -> {out.name}")


def download_tinystories(lang: str) -> None:
    out = RAW_DATA_DIR / f"{lang}_tinystories.txt"
    if out.exists():
        print(f"  [{lang}] tinystories already exists, skipping")
        return
    print(f"  [{lang}] Downloading TinyStories...")
    ds = load_dataset(
        "deeponh/multilingual-tinystories",
        "default",
        split=TINYSTORIES_SPLITS[lang],
        streaming=True,
    )
    written = 0
    with out.open("w", encoding="utf-8") as f:
        for row in tqdm(ds, desc=f"{lang}/tinystories", unit="lines"):
            text = (row.get("text") or row.get("story") or "").strip()
            if text:
                f.write(text + "\n")
                written += 1
    print(f"  [{lang}] tinystories done: {written:,} lines -> {out.name}")


DOWNLOADERS = {
    "cc100": download_cc100,
    "wiki": download_wiki,
    "samanantar": download_samanantar,
    "tinystories": download_tinystories,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download raw corpora for Dravidian LM research.")
    parser.add_argument("--lang", choices=LANGUAGES, default=None)
    parser.add_argument("--source", choices=list(DOWNLOADERS), default=None)
    args = parser.parse_args()

    langs = [args.lang] if args.lang else LANGUAGES
    sources = [args.source] if args.source else list(DOWNLOADERS)

    print(f"Downloading: langs={langs} sources={sources}")
    print(f"Output dir: {RAW_DATA_DIR}\n")

    for lang in langs:
        for src in sources:
            try:
                DOWNLOADERS[src](lang)
            except Exception as exc:
                print(f"  [{lang}] {src} FAILED: {exc}")

    print("\nAll downloads complete.")


if __name__ == "__main__":
    main()
