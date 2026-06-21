import argparse
import os
import random
from pathlib import Path

import sentencepiece as spm
from datasets import load_dataset

BASE = Path("/nfs/storage1/home/pulipakv/Dravidian")
PROC_DIR = BASE / "data" / "processed"
TOK_DIR = BASE / "tokenizers"
HF_DATASET_REPO = os.environ.get("DRAVIDIAN_DATASET", "pulipakav-1/dravidian")

LANGUAGES = ["te", "ta", "kn", "ml"]
LANGUAGE_NAMES = {"te": "telugu", "ta": "tamil", "kn": "kannada", "ml": "malayalam"}
MONO_VOCAB_SIZE = 32_000
JOINT_VOCAB_SIZE = 64_000
NORM_RULE = "nmt_nfkc"

# 100M words per language -> 400M total for joint tokenizer.
JOINT_SAMPLE_WORDS = 100_000_000


def _spm_train(input_path: str, prefix: str, vocab_size: int = MONO_VOCAB_SIZE) -> None:
    spm.SentencePieceTrainer.train(
        input=input_path,
        model_prefix=prefix,
        model_type="bpe",
        vocab_size=vocab_size,
        character_coverage=1.0,
        normalization_rule_name=NORM_RULE,
        byte_fallback=True,
        num_threads=8,
        shuffle_input_sentence=True,
        train_extremely_large_corpus=True,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
    )


def load_local_lines(lang: str) -> list[str]:
    in_path = PROC_DIR / f"{lang}_train.txt"
    if not in_path.exists():
        raise FileNotFoundError(in_path)
    return [line.strip() for line in in_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def extract_texts(rows) -> list[str]:
    texts: list[str] = []
    for row in rows:
        text = (
            row.get("text")
            or row.get("content")
            or row.get("sentence")
            or row.get("story")
            or ""
        )
        text = text.strip()
        if text:
            texts.append(text)
    return texts


def load_hf_lines(lang: str) -> list[str]:
    language_name = LANGUAGE_NAMES[lang]
    attempts = [
        {"data_dir": language_name, "split": "train"},
        {"name": language_name, "split": "train"},
    ]
    errors: list[str] = []

    for kwargs in attempts:
        try:
            rows = load_dataset(HF_DATASET_REPO, **kwargs)
            texts = extract_texts(rows)
            if texts:
                return texts
            errors.append(f"{kwargs} -> no usable text rows")
        except Exception as exc:
            errors.append(f"{kwargs} -> {exc}")

    raise ValueError(
        f"Unable to load tokenizer training text from dataset={HF_DATASET_REPO} "
        f"for language={language_name}. Attempts: {'; '.join(errors)}"
    )


def load_train_lines(lang: str) -> list[str]:
    try:
        return load_hf_lines(lang)
    except Exception as dataset_exc:
        print(f"  [{lang}] falling back to local processed train file: {dataset_exc}")
        return load_local_lines(lang)


def write_training_text(lines: list[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_mono(lang: str) -> None:
    out_dir = TOK_DIR / lang
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(out_dir / "tokenizer")
    temp_input = out_dir / f"{lang}_train_for_tokenizer.txt"

    if (out_dir / "tokenizer.model").exists():
        print(f"  [{lang}] tokenizer already exists - skipping")
        return

    try:
        lines = load_train_lines(lang)
    except Exception as exc:
        print(f"  [{lang}] failed to load training text - skipping: {exc}")
        return

    print(f"  [{lang}] Training monolingual tokenizer")
    write_training_text(lines, temp_input)
    try:
        _spm_train(str(temp_input), prefix)
    finally:
        if temp_input.exists():
            temp_input.unlink()
    print(f"  [{lang}] Done -> {out_dir}/tokenizer.model")


def sample_joint_lines(lang: str, max_words: int) -> tuple[list[str], int]:
    lines = load_train_lines(lang)
    words = 0
    sampled: list[str] = []
    for line in lines:
        sampled.append(line)
        words += len(line.split())
        if words >= max_words:
            break
    random.shuffle(sampled)
    return sampled, words


def train_joint() -> None:
    out_dir = TOK_DIR / "joint"
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(out_dir / "tokenizer")
    joint_input = out_dir / "joint_input.txt"

    if (out_dir / "tokenizer.model").exists():
        print("  [joint] tokenizer already exists - skipping")
        return

    print(f"  [joint] Sampling {JOINT_SAMPLE_WORDS:,} words per language...")
    wrote_any = False
    with open(joint_input, "w", encoding="utf-8") as fout:
        for lang in LANGUAGES:
            try:
                lines, words = sample_joint_lines(lang, JOINT_SAMPLE_WORDS)
            except Exception as exc:
                print(f"  [joint] failed to load {lang} - skipping: {exc}")
                continue

            for line in lines:
                fout.write(line + "\n")
            wrote_any = True
            print(f"  [joint]   {lang}: {words:,} words / {len(lines):,} lines")

    if not wrote_any:
        if joint_input.exists():
            joint_input.unlink()
        raise RuntimeError("No training text was available for the joint tokenizer.")

    print("  [joint] Training joint tokenizer...")
    try:
        _spm_train(str(joint_input), prefix, vocab_size=JOINT_VOCAB_SIZE)
    finally:
        if joint_input.exists():
            joint_input.unlink()
    print(f"  [joint] Done -> {out_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=LANGUAGES, default=None)
    parser.add_argument("--joint-only", action="store_true")
    args = parser.parse_args()

    print("Training tokenizers")
    print(f"  Dataset       : {HF_DATASET_REPO}")
    print(f"  Vocab size    : {MONO_VOCAB_SIZE:,} (mono) / {JOINT_VOCAB_SIZE:,} (joint)")
    print("  Coverage      : 1.0")
    print(f"  Normalization : {NORM_RULE}")
    print("  Type          : BPE + byte fallback")
    print("  Large corpus  : True\n")

    if not args.joint_only:
        langs = [args.lang] if args.lang else LANGUAGES
        for lang in langs:
            train_mono(lang)

    if args.lang is None:
        train_joint()

    print("\nAll tokenizers trained.")


if __name__ == "__main__":
    main()
