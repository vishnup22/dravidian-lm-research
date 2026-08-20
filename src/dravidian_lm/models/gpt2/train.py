from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

from datasets import Dataset, load_from_disk
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2LMHeadModel,
    T5Tokenizer,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a monolingual GPT-2 language model.")
    parser.add_argument("--language", required=True, help="Language name or code.")
    parser.add_argument(
        "--tokenizer_name",
        required=True,
        help="Tokenizer directory under artifacts/tokenizers/.",
    )
    parser.add_argument(
        "--variant",
        default="full",
        choices=["full", "easy", "hard", "mid", "random"],
        help=(
            "Which pruned training split to use (data/splits/{code}/pruned/"
            "{code}_train_{variant}.txt). 'full' (default) trains on the "
            "unpruned split, matching prior behavior exactly."
        ),
    )
    parser.add_argument(
        "--eval_strategy",
        default="epoch",
        choices=["epoch", "steps"],
        help="'steps' logs eval_loss every --eval_steps, for tokens-vs-BPB curves.",
    )
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Run a single seed instead of the default START_SEED..START_SEED+NUM_SEEDS-1 sweep.",
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


def load_tokenizer(tokenizer_name: str) -> T5Tokenizer:
    model_file = TOKENIZERS_DIR / tokenizer_name / "tokenizer.model"
    if not model_file.exists():
        raise FileNotFoundError(f"Tokenizer model not found: {model_file}")
    tok = T5Tokenizer(vocab_file=str(model_file), extra_ids=0)
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


def split_path(language_code: str, split: str, variant: str = "full") -> Path:
    if split == "train" and variant != "full":
        return SPLITS_DIR / language_code / "pruned" / f"{language_code}_train_{variant}.txt"
    return SPLITS_DIR / language_code / f"{language_code}_{split}.txt"


def build_or_load_tokenized_dataset(
    language_code: str, split: str, tokenizer: T5Tokenizer, variant: str = "full"
) -> Dataset:
    # val/test caches are shared across every variant (same source file) so we
    # only tag the cache name when it's the pruned train split being loaded.
    variant_suffix = f"_{variant}" if split == "train" and variant != "full" else ""
    cache_name = f"{language_code}_{split}{variant_suffix}_tokenized_ctx{MAX_LENGTH}_tok32k"
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

    src = split_path(language_code, split, variant)
    if not src.exists():
        raise FileNotFoundError(f"Missing {split} split for {language_code}: {src}")

    texts = load_lines(src)
    rng = random.Random(1)
    rng.shuffle(texts)

    enc = tokenizer(
        texts,
        truncation=True,
        padding=False,
        max_length=MAX_LENGTH,
        add_special_tokens=False,
    )
    ds = Dataset.from_dict(
        {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }
    )
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


def bytes_per_token(language_code: str, tokenizer: T5Tokenizer) -> float:
    """Average UTF-8 bytes per token over the (shared) val split, used to convert
    eval_loss (nats/token) into bits-per-byte for the tokens-vs-BPB curve."""
    val_texts = load_lines(split_path(language_code, "val"))
    enc = tokenizer(val_texts, truncation=True, max_length=MAX_LENGTH, add_special_tokens=False)
    total_bytes = sum(len(t.encode("utf-8")) for t in val_texts)
    total_tokens = sum(len(ids) for ids in enc["input_ids"])
    return total_bytes / max(total_tokens, 1)


def build_eval_history(log_history: list[dict], tokens_per_step: float, bpb_ratio: float) -> list[dict]:
    history = []
    for entry in log_history:
        if "eval_loss" not in entry:
            continue
        eval_loss = entry["eval_loss"]
        step = entry.get("step", 0)
        history.append(
            {
                "step": step,
                "epoch": entry.get("epoch"),
                "tokens_seen": round(step * tokens_per_step),
                "eval_loss": eval_loss,
                "bpb": round((eval_loss / math.log(2)) / bpb_ratio, 6) if bpb_ratio else None,
            }
        )
    return history


def train_one(
    language: str,
    seed: int,
    tokenizer_name: str,
    variant: str = "full",
    eval_strategy: str = "epoch",
    eval_steps: int = 500,
) -> None:
    language_name, language_code = normalize_language(language)
    run_name = f"seed{seed}" if variant == "full" else f"seed{seed}_{variant}"
    output_dir = MODELS_DIR / "gpt2" / language_name / run_name
    RAW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RAW_RESULTS_DIR / f"{language_name}_{run_name}.json"

    rank = int(os.environ.get("RANK", "0"))

    if output_dir.exists() and log_path.exists():
        if rank == 0:
            print(f"[{time.strftime('%H:%M:%S')}] skipping {language_name}/{run_name} - already done", flush=True)
        return

    if rank == 0:
        print(f"[{time.strftime('%H:%M:%S')}] starting {language_name}/{run_name} (variant={variant})", flush=True)
    set_seed(seed)

    tokenizer = load_tokenizer(tokenizer_name)
    train_dataset = build_or_load_tokenized_dataset(language_code, "train", tokenizer, variant)
    val_dataset = build_or_load_tokenized_dataset(language_code, "val", tokenizer, variant)
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
    avg_seq_len = sum(len(ids) for ids in train_dataset["input_ids"]) / max(len(train_dataset), 1)
    tokens_per_step = PER_DEVICE_BATCH * GRAD_ACCUM * max(1, world_size) * avg_seq_len
    if rank == 0:
        print(
            f"[{time.strftime('%H:%M:%S')}] world_size={world_size}, "
            f"dataset={len(train_dataset):,}, estimated_total_steps={total_steps:,}, "
            f"avg_seq_len={avg_seq_len:.1f}",
            flush=True,
        )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=run_name,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        per_device_eval_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=N_EPOCHS,
        evaluation_strategy=eval_strategy,
        eval_steps=eval_steps,
        logging_steps=200,
        save_strategy=eval_strategy,
        save_steps=eval_steps,
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

    # If a previous (e.g. timed-out) run left checkpoints in output_dir but never
    # finished (no log_path -> the skip check above didn't trigger), resume from
    # the latest one instead of retraining from scratch.
    has_checkpoint = output_dir.exists() and any(output_dir.glob("checkpoint-*"))
    if rank == 0 and has_checkpoint:
        print(f"[{time.strftime('%H:%M:%S')}] resuming {language_name}/{run_name} from latest checkpoint", flush=True)

    start = time.time()
    train_result = trainer.train(resume_from_checkpoint=has_checkpoint)
    elapsed = time.time() - start
    eval_result = trainer.evaluate()
    trainer.save_model(str(output_dir))

    eval_loss = eval_result.get("eval_loss")
    perplexity = math.exp(eval_loss) if eval_loss is not None else None

    if rank == 0:
        bpb_ratio = bytes_per_token(language_code, tokenizer)
        eval_history = build_eval_history(trainer.state.log_history, tokens_per_step, bpb_ratio)

        log = {
            "language": language_name,
            "language_code": language_code,
            "architecture": "gpt2",
            "tokenizer_name": tokenizer_name,
            "run_name": run_name,
            "variant": variant,
            "seed": seed,
            "num_epochs": N_EPOCHS,
            "total_steps": total_steps,
            "tokens_per_step": tokens_per_step,
            "train_runtime_seconds": elapsed,
            "train_loss": train_result.training_loss,
            "eval_loss": eval_loss,
            "perplexity": perplexity,
            "train_metrics": train_result.metrics,
            "eval_metrics": eval_result,
            "eval_history": eval_history,
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
    seeds = [args.seed] if args.seed is not None else range(START_SEED, START_SEED + NUM_SEEDS)
    for seed in seeds:
        print(f"=== {args.language.upper()} seed={seed} variant={args.variant} ===", flush=True)
        train_one(
            args.language,
            seed,
            args.tokenizer_name,
            variant=args.variant,
            eval_strategy=args.eval_strategy,
            eval_steps=args.eval_steps,
        )
    print(f"=== {args.language.upper()} DONE ===", flush=True)


if __name__ == "__main__":
    main()
