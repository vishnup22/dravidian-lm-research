from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_from_disk
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2LMHeadModel,
    T5TokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)

from dravidian_lm.paths import MODELS_DIR, RAW_RESULTS_DIR, SPLITS_DIR, TOKENIZERS_DIR


os.environ["TRANSFORMERS_NO_FLASH_ATTN"] = "1"

START_SEED = 1
NUM_SEEDS = 2

MAX_LENGTH = 1024
PER_DEVICE_BATCH = 4
GRAD_ACCUM = 16
N_EPOCHS = 3
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 4000
MAX_GRAD_NORM = 0.5
NUM_WORKERS = 2

# Parallelize the one-time tokenization/caching pass across CPU cores instead of
# encoding tens of millions of lines in a single process. Match --cpus-per-task
# in the SLURM launch scripts.
TOKENIZE_NUM_PROC = int(os.environ.get("SLURM_CPUS_PER_TASK", "16"))

LANGUAGE_CODES = {
    "telugu": "te",
    "tamil": "ta",
    "kannada": "kn",
    "malayalam": "ml",
    "te": "te",
    "ta": "ta",
    "kn": "kn",
    "ml": "ml",
}

ALL_LANGUAGE_CODES = ["te", "ta", "kn", "ml"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GPT-2 language model.")
    parser.add_argument(
        "--language",
        required=True,
        help="Language name or code, or 'multilingual' to train jointly on all four languages.",
    )
    parser.add_argument(
        "--tokenizer_name",
        required=True,
        help="Tokenizer directory under artifacts/tokenizers/.",
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=NUM_SEEDS,
        help=f"Number of seeds to run (default: {NUM_SEEDS}).",
    )
    parser.add_argument(
        "--start_seed",
        type=int,
        default=START_SEED,
        help=f"First seed to run (default: {START_SEED}).",
    )
    return parser.parse_args()


def build_model(vocab_size: int) -> GPT2LMHeadModel:
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=3072,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    )
    return GPT2LMHeadModel(config)


def normalize_language(language: str) -> tuple[str, str]:
    key = language.lower()
    if key not in LANGUAGE_CODES:
        raise ValueError(f"Unsupported language: {language}")
    code = LANGUAGE_CODES[key]
    full_name = next(
        name for name, lang_code in LANGUAGE_CODES.items() if len(name) > 2 and lang_code == code
    )
    return full_name, code


def load_tokenizer(tokenizer_name: str) -> T5TokenizerFast:
    model_file = TOKENIZERS_DIR / tokenizer_name / "tokenizer.model"
    if not model_file.exists():
        raise FileNotFoundError(f"Tokenizer model not found: {model_file}")
    tok = T5TokenizerFast(vocab_file=str(model_file), extra_ids=0)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "<pad>"})
    return tok


def load_lines(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def wait_for_cache_ready(cache_path: Path, ready_path: Path, timeout_s: int = 7200) -> None:
    start = time.time()
    while True:
        if cache_path.exists() and ready_path.exists():
            return
        if time.time() - start > timeout_s:
            raise TimeoutError(
                f"Timed out waiting for tokenized cache: {cache_path} (ready={ready_path})"
            )
        time.sleep(2)


def split_path(language_code: str, split: str) -> Path:
    return SPLITS_DIR / language_code / f"{language_code}_{split}.txt"


def build_or_load_tokenized_dataset(
    language_code: str, split: str, tokenizer: T5TokenizerFast, tokenizer_name: str
) -> Dataset:
    cache_name = f"{language_code}_{split}_tokenized_ctx{MAX_LENGTH}_{tokenizer_name}"
    cache_path = SPLITS_DIR / "cache" / cache_name
    ready_path = SPLITS_DIR / "cache" / f"{cache_name}.ready"

    if cache_path.exists():
        return load_from_disk(str(cache_path))

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1 and rank != 0:
        print(
            f"[{time.strftime('%H:%M:%S')}] rank={rank} waiting for rank0 to build {cache_name}",
            flush=True,
        )
        wait_for_cache_ready(cache_path, ready_path)
        return load_from_disk(str(cache_path))

    src = split_path(language_code, split)
    if not src.exists():
        raise FileNotFoundError(f"Missing {split} split for {language_code}: {src}")

    texts = load_lines(src)
    rng = random.Random(1)
    rng.shuffle(texts)

    def _encode(batch: dict) -> dict:
        return tokenizer(
            batch["text"],
            truncation=True,
            padding=False,
            max_length=MAX_LENGTH,
            add_special_tokens=False,
        )

    raw_ds = Dataset.from_dict({"text": texts})
    ds = raw_ds.map(
        _encode,
        batched=True,
        num_proc=TOKENIZE_NUM_PROC,
        remove_columns=["text"],
        desc=f"tokenizing {language_code}/{split}",
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(cache_path))
    ready_path.write_text("ok\n", encoding="utf-8")
    return ds


def build_or_load_multilingual_dataset(
    split: str, tokenizer: T5TokenizerFast, tokenizer_name: str
) -> Dataset:
    """Concatenate the per-language tokenized datasets for a joint/multilingual run."""
    cache_name = f"multilingual_{split}_tokenized_ctx{MAX_LENGTH}_{tokenizer_name}"
    cache_path = SPLITS_DIR / "cache" / cache_name
    ready_path = SPLITS_DIR / "cache" / f"{cache_name}.ready"

    if cache_path.exists():
        return load_from_disk(str(cache_path))

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1 and rank != 0:
        print(
            f"[{time.strftime('%H:%M:%S')}] rank={rank} waiting for rank0 to build {cache_name}",
            flush=True,
        )
        wait_for_cache_ready(cache_path, ready_path)
        return load_from_disk(str(cache_path))

    parts = [
        build_or_load_tokenized_dataset(lang_code, split, tokenizer, tokenizer_name)
        for lang_code in ALL_LANGUAGE_CODES
    ]
    ds = concatenate_datasets(parts).shuffle(seed=1)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(cache_path))
    ready_path.write_text("ok\n", encoding="utf-8")
    return ds


def compute_total_steps(
    dataset_size: int,
    batch_size: int,
    grad_accum: int,
    num_gpus: int,
    num_epochs: int,
) -> int:
    steps_per_epoch = max(1, dataset_size // (batch_size * grad_accum * max(1, num_gpus)))
    return steps_per_epoch * num_epochs


def train_one(language: str, seed: int, tokenizer_name: str) -> None:
    if language.lower() == "multilingual":
        language_name, language_code = "multilingual", "joint"
    else:
        language_name, language_code = normalize_language(language)
    run_name = f"seed{seed}"
    output_dir = MODELS_DIR / "gpt2" / language_name / run_name
    RAW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RAW_RESULTS_DIR / f"{language_name}_{run_name}.json"

    rank = int(os.environ.get("RANK", "0"))

    if output_dir.exists() and log_path.exists():
        if rank == 0:
            print(f"[{time.strftime('%H:%M:%S')}] skipping {language_name}/{run_name} - already done", flush=True)
        return

    if rank == 0:
        print(f"[{time.strftime('%H:%M:%S')}] starting {language_name}/{run_name}", flush=True)
    set_seed(seed)

    tokenizer = load_tokenizer(tokenizer_name)
    if language_code == "joint":
        train_dataset = build_or_load_multilingual_dataset("train", tokenizer, tokenizer_name)
        val_dataset = build_or_load_multilingual_dataset("val", tokenizer, tokenizer_name)
    else:
        train_dataset = build_or_load_tokenized_dataset(language_code, "train", tokenizer, tokenizer_name)
        val_dataset = build_or_load_tokenized_dataset(language_code, "val", tokenizer, tokenizer_name)
    model = build_model(vocab_size=len(tokenizer))
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    total_steps = compute_total_steps(
        dataset_size=len(train_dataset),
        batch_size=PER_DEVICE_BATCH,
        grad_accum=GRAD_ACCUM,
        num_gpus=max(1, world_size),
        num_epochs=N_EPOCHS,
    )
    if rank == 0:
        print(
            f"[{time.strftime('%H:%M:%S')}] world_size={world_size}, "
            f"dataset={len(train_dataset):,}, estimated_total_steps={total_steps:,}",
            flush=True,
        )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=run_name,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        per_device_eval_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=N_EPOCHS,
        evaluation_strategy="epoch",
        logging_steps=200,
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        lr_scheduler_type="cosine",
        max_grad_norm=MAX_GRAD_NORM,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        dataloader_num_workers=NUM_WORKERS,
        seed=seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    start = time.time()
    train_result = trainer.train()
    elapsed = time.time() - start
    eval_result = trainer.evaluate()
    trainer.save_model(str(output_dir))

    eval_loss = eval_result.get("eval_loss")
    perplexity = math.exp(eval_loss) if eval_loss is not None else None

    if rank == 0:
        log = {
            "language": language_name,
            "language_code": language_code,
            "architecture": "gpt2",
            "tokenizer_name": tokenizer_name,
            "run_name": run_name,
            "seed": seed,
            "num_epochs": N_EPOCHS,
            "total_steps": total_steps,
            "train_runtime_seconds": elapsed,
            "train_loss": train_result.training_loss,
            "eval_loss": eval_loss,
            "perplexity": perplexity,
            "train_metrics": train_result.metrics,
            "eval_metrics": eval_result,
        }
        with log_path.open("w", encoding="utf-8") as f:
            json.dump(log, f, indent=2, ensure_ascii=False)

        if eval_loss is not None and perplexity is not None:
            print(
                f"[{time.strftime('%H:%M:%S')}] done {language_name}/{run_name} - "
                f"eval_loss={eval_loss:.4f}, perplexity={perplexity:.2f}",
                flush=True,
            )


def main() -> None:
    args = parse_args()
    for seed in range(args.start_seed, args.start_seed + args.num_seeds):
        print(f"=== {args.language.upper()} seed={seed} ===", flush=True)
        train_one(args.language, seed, args.tokenizer_name)
    print(f"=== {args.language.upper()} DONE ===", flush=True)


if __name__ == "__main__":
    main()
