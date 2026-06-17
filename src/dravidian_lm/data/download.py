#!/usr/bin/env python3

from __future__ import annotations

import argparse
import lzma
import urllib.request

from datasets import load_dataset
from tqdm import tqdm

from dravidian_lm.paths import RAW_DATA_DIR


LANGUAGES = ["te", "ta", "kn", "ml"]

# CC100 is no longer available via the HuggingFace dataset hub (loading scripts
# were deprecated). Download the original xz files from statmt.org instead.
CC100_URLS = {
    "te": "http://data.statmt.org/cc-100/te.txt.xz",
    "ta": "http://data.statmt.org/cc-100/ta.txt.xz",
    "kn": "http://data.statmt.org/cc-100/kn.txt.xz",
    "ml": "http://data.statmt.org/cc-100/ml.txt.xz",
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


class _DownloadProgress(tqdm):
    def update_to(self, b: int = 1, bsize: int = 1, tsize: int | None = None) -> None:
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_cc100(lang: str) -> None:
    out = RAW_DATA_DIR / f"{lang}_cc100.txt"
    if out.exists():
        print(f"  [{lang}] cc100 already exists, skipping")
        return
    url = CC100_URLS[lang]
    xz_path = RAW_DATA_DIR / f"{lang}_cc100.txt.xz"
    print(f"  [{lang}] Downloading CC100 from {url} ...")
    with _DownloadProgress(unit="B", unit_scale=True, miniters=1, desc=f"{lang}/cc100.xz") as pbar:
        urllib.request.urlretrieve(url, xz_path, reporthook=pbar.update_to)
    print(f"  [{lang}] Extracting...")
    written = 0
    with lzma.open(xz_path, "rt", encoding="utf-8") as fin:
        with out.open("w", encoding="utf-8") as fout:
            for line in tqdm(fin, desc=f"{lang}/cc100", unit="lines"):
                text = line.strip()
                if text:
                    fout.write(text + "\n")
                    written += 1
    xz_path.unlink()
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
