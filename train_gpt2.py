from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

from datasets import Dataset, load_dataset, load_from_disk
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2LMHeadModel,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

os.environ["TRANSFORMERS_NO_FLASH_ATTN"] = "1"

ROOT = Path.cwd()
ENV_BASE = os.environ.get("DRAVIDIAN_BASE")
DEFAULT_CLUSTER_BASE = Path("/nfs/storage1/home/pulipakv/Dravidian")
HF_DATASET_REPO = os.environ.get("DRAVIDIAN_DATASET", "pulipakav-1/dravidian")

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

LANGUAGE_NAMES = {
    "te": "telugu",
    "ta": "tamil",
    "kn": "kannada",
    "ml": "malayalam",
}


def candidate_roots() -> list[Path]:
    roots = [ROOT]
    if ENV_BASE:
        roots.append(Path(ENV_BASE))
    if DEFAULT_CLUSTER_BASE not in roots:
        roots.append(DEFAULT_CLUSTER_BASE)
    return roots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GPT-2 monolingual language model")
    parser.add_argument(
        "--language",
        required=True,
        help="Language to train (telugu, tamil, kannada, malayalam or te, ta, kn, ml)",
    )
    parser.add_argument(
        "--tokenizer_dirname",
        required=True,
        help="Tokenizer directory name under tokenizers/",
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
    key = language.strip().lower()
    if key not in LANGUAGE_CODES:
        raise ValueError(
            f"Unsupported language '{language}'. Expected one of: "
            f"{', '.join(sorted(set(LANGUAGE_CODES)))}"
        )

    language_code = LANGUAGE_CODES[key]
    return LANGUAGE_NAMES[language_code], language_code


def resolve_tokenizer_model(language: str, tokenizer_dirname: str) -> Path:
    language_name, language_code = normalize_language(language)
    candidates: list[Path] = []
    for base in candidate_roots():
        candidates.extend(
            [
                base / "tokenizers" / tokenizer_dirname / f"{language_name}.model",
                base / "tokenizers" / tokenizer_dirname / f"{language_code}.model",
                base / "tokenizers" / tokenizer_dirname / "tokenizer.model",
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Tokenizer model not found. Tried:\n"
        + "\n".join(f"  - {path}" for path in candidates)
    )


def load_tokenizer(language: str, tokenizer_dirname: str) -> T5Tokenizer:
    model_file = resolve_tokenizer_model(language, tokenizer_dirname)
    tok = T5Tokenizer(vocab_file=str(model_file), extra_ids=0)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "<pad>"})
    return tok


def load_lines(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip()]


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


def load_hf_split(language: str, split: str) -> list[str]:
    language_name, _ = normalize_language(language)
    attempts = [
        {"data_dir": language_name, "split": split},
        {"name": language_name, "split": split},
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
        f"Unable to load dataset={HF_DATASET_REPO} for language={language_name}, "
        f"split={split}. Attempts: {'; '.join(errors)}"
    )


def resolve_split_source(language: str, split: str) -> Path:
    language_name, language_code = normalize_language(language)
    split_candidates = {"train": [], "val": [], "test": []}
    for base in candidate_roots():
        split_candidates["train"].extend(
            [
                base / "clean_data" / language_name / "train" / f"{language_name}_train_balanced.txt",
                base / "clean_data" / language_code / "train" / f"{language_code}_train_balanced.txt",
                base / "data" / "processed" / f"{language_code}_train.txt",
                base / "data" / "processed" / f"{language_name}_train.txt",
            ]
        )
        split_candidates["val"].extend(
            [
                base / "clean_data" / language_name / "val" / f"{language_name}_val_cleaned_final.txt",
                base / "clean_data" / language_code / "val" / f"{language_code}_val_cleaned_final.txt",
                base / "data" / "processed" / f"{language_code}_val.txt",
                base / "data" / "processed" / f"{language_name}_val.txt",
            ]
        )
        split_candidates["test"].extend(
            [
                base / "clean_data" / language_name / "test" / f"{language_name}_test_cleaned_final.txt",
                base / "clean_data" / language_code / "test" / f"{language_code}_test_cleaned_final.txt",
                base / "data" / "processed" / f"{language_code}_test.txt",
                base / "data" / "processed" / f"{language_name}_test.txt",
            ]
        )

    if split not in split_candidates:
        raise ValueError(f"Unsupported split: {split}")

    for candidate in split_candidates[split]:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Missing cleaned {split} file for {language}. Tried:\n"
        + "\n".join(f"  - {path}" for path in split_candidates[split])
    )


def wait_for_cache_ready(cache_path: Path, ready_path: Path, timeout_s: int = 7200) -> None:
    error_path = Path(str(ready_path).replace(".ready", ".error"))
    start = time.time()
    while True:
        if cache_path.exists() and ready_path.exists():
            return
        if error_path.exists():
            raise RuntimeError(
                f"Rank 0 failed to build tokenized cache: {cache_path}\n"
                f"See {error_path} for details."
            )
        if time.time() - start > timeout_s:
            raise TimeoutError(
                f"Timed out waiting for tokenized cache: {cache_path} (ready={ready_path})"
            )
        time.sleep(2)


def build_or_load_tokenized_dataset(language: str, split: str, tokenizer: T5Tokenizer) -> Dataset:
    cache_name = f"{language}_{split}_tokenized_ctx{MAX_LENGTH}_tok32k"
    cache_path = ROOT / "data" / cache_name
    ready_path = ROOT / "data" / f"{cache_name}.ready"

    if cache_path.exists() and ready_path.exists():
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

    error_path = cache_path.parent / f"{cache_name}.error"

    try:
        try:
            texts = load_hf_split(language, split)
        except Exception as dataset_exc:
            src = resolve_split_source(language, split)
            print(
                f"[{time.strftime('%H:%M:%S')}] falling back to local {split} data for {language}: {dataset_exc}",
                flush=True,
            )
            texts = load_lines(src)
        rng = random.Random(1)
        rng.shuffle(texts)

        enc = tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=MAX_LENGTH,
            add_special_tokens=False,  # suppress T5 EOS so CLM labels stay aligned
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
    except Exception as exc:
        error_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        raise


def compute_total_steps(
    dataset_size: int, batch_size: int, grad_accum: int, num_gpus: int, num_epochs: int
) -> int:
    steps_per_epoch = max(1, dataset_size // (batch_size * grad_accum * max(1, num_gpus)))
    return steps_per_epoch * num_epochs


def train_one(language: str, seed: int, tokenizer_dirname: str) -> None:
    run_name = f"seed{seed}"
    output_dir = ROOT / "models" / language / run_name
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / f"{language}_{run_name}.json"

    rank = int(os.environ.get("RANK", "0"))

    if output_dir.exists() and log_path.exists():
        if rank == 0:
            print(f"[{time.strftime('%H:%M:%S')}] skipping {language}/{run_name} - already done", flush=True)
        return

    if rank == 0:
        print(f"[{time.strftime('%H:%M:%S')}] starting {language}/{run_name}", flush=True)
    set_seed(seed)

    tokenizer = load_tokenizer(language, tokenizer_dirname)
    train_dataset = build_or_load_tokenized_dataset(language, "train", tokenizer)
    val_dataset = build_or_load_tokenized_dataset(language, "val", tokenizer)
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
            "language": language,
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
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2, ensure_ascii=False)

        print(
            f"[{time.strftime('%H:%M:%S')}] done {language}/{run_name} - "
            f"eval_loss={eval_loss:.4f}, perplexity={perplexity:.2f}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    language = args.language
    tokenizer_dirname = args.tokenizer_dirname

    for seed in range(START_SEED, START_SEED + NUM_SEEDS):
        print(f"=== {language.upper()} seed={seed} ===", flush=True)
        train_one(language, seed, tokenizer_dirname)

    print(f"=== {language.upper()} DONE ===", flush=True)


if __name__ == "__main__":
    main()
