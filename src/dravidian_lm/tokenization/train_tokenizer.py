from __future__ import annotations

import argparse
import random

import sentencepiece as spm

from dravidian_lm.paths import SPLITS_DIR, TOKENIZERS_DIR


LANGUAGES = ["te", "ta", "kn", "ml"]
MONO_VOCAB_SIZE = 32_000
JOINT_VOCAB_SIZE = 64_000
NORM_RULE = "nmt_nfkc"
JOINT_SAMPLE_WORDS = 100_000_000

# Without a cap, SentencePiece loads the entire input corpus into memory before
# training. Tamil and Malayalam CC-100 + wiki + samanantar corpora run into the
# tens of GB, which OOMs. Capping input_sentence_size (paired with
# shuffle_input_sentence) makes SentencePiece randomly subsample this many
# sentences instead of reading everything.
SPM_INPUT_SENTENCE_SIZE = 10_000_000


def spm_train(input_path: str, prefix: str, vocab_size: int = MONO_VOCAB_SIZE) -> None:
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
        input_sentence_size=SPM_INPUT_SENTENCE_SIZE,
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


def train_mono(lang: str) -> None:
    in_path = SPLITS_DIR / lang / f"{lang}_train.txt"
    out_dir = TOKENIZERS_DIR / lang
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(out_dir / "tokenizer")

    if (out_dir / "tokenizer.model").exists():
        print(f"  [{lang}] tokenizer already exists, skipping")
        return
    if not in_path.exists():
        print(f"  [{lang}] {in_path.name} not found, skipping")
        return

    print(f"  [{lang}] Training monolingual tokenizer...")
    spm_train(str(in_path), prefix)
    print(f"  [{lang}] Done -> {out_dir / 'tokenizer.model'}")


def train_joint() -> None:
    out_dir = TOKENIZERS_DIR / "joint"
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(out_dir / "tokenizer")
    joint_input = out_dir / "joint_input.txt"

    if (out_dir / "tokenizer.model").exists():
        print("  [joint] tokenizer already exists, skipping")
        return

    print(f"  [joint] Sampling {JOINT_SAMPLE_WORDS:,} words per language...")
    with joint_input.open("w", encoding="utf-8") as fout:
        for lang in LANGUAGES:
            in_path = SPLITS_DIR / lang / f"{lang}_train.txt"
            if not in_path.exists():
                print(f"  [joint] {in_path.name} not found, skipping {lang}")
                continue

            words = 0
            lines: list[str] = []
            with in_path.open(encoding="utf-8", errors="ignore") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    lines.append(line)
                    words += len(line.split())
                    if words >= JOINT_SAMPLE_WORDS:
                        break

            random.shuffle(lines)
            for line in lines:
                fout.write(line + "\n")
            print(f"  [joint]   {lang}: {words:,} words / {len(lines):,} lines")

    print("  [joint] Training joint tokenizer...")
    spm_train(str(joint_input), prefix, vocab_size=JOINT_VOCAB_SIZE)
    joint_input.unlink()
    print(f"  [joint] Done -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SentencePiece tokenizers for Dravidian corpora.")
    parser.add_argument("--lang", choices=LANGUAGES, default=None)
    parser.add_argument("--joint-only", action="store_true")
    args = parser.parse_args()

    print("=" * 62)
    print("  TOKENIZER TRAINING")
    print("=" * 62)
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
