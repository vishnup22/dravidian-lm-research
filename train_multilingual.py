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
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


os.environ["TRANSFORMERS_NO_FLASH_ATTN"] = "1"


ROOT = Path.cwd()


START_SEED = 1
NUM_SEEDS = 1


MAX_LENGTH = 1024
PER_DEVICE_BATCH = 4
GRAD_ACCUM = 16
N_EPOCHS = 3
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 4000
MAX_GRAD_NORM = 0.5
NUM_WORKERS = 2


ALL_LANGUAGES = ["te", "ta", "kn", "ml"]




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GPT-2 multilingual model on Dravidian languages.")
    parser.add_argument(
        "--languages",
        default=",".join(ALL_LANGUAGES),
        help="Comma-separated language codes to include (default: te,ta,kn,ml).",
    )
    parser.add_argument(
        "--tokenizer_dirname",
        default="joint",
        help="Tokenizer directory name under tokenizers/ (default: joint).",
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
        bos_token_id=2,
        eos_token_id=3,
    )
    return GPT2LMHeadModel(config)




def load_tokenizer(tokenizer_dirname: str) -> T5Tokenizer:
    model_file = ROOT / "tokenizers" / tokenizer_dirname / "tokenizer.model"
    if not model_file.exists():
        raise FileNotFoundError(f"Joint tokenizer not found: {model_file}")
    tok = T5Tokenizer(vocab_file=str(model_file), extra_ids=0)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "<pad>"})
    return tok




def load_lines(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip()]




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
            raise TimeoutError(f"Timed out waiting for tokenized cache: {cache_path}")
        time.sleep(2)




def build_or_load_language_dataset(
    language: str, split: str, tokenizer: T5Tokenizer
) -> Dataset:
    cache_name = f"{language}_{split}_tokenized_ctx{MAX_LENGTH}_tok64k"
    cache_path = ROOT / "data" / cache_name
    ready_path = ROOT / "data" / f"{cache_name}.ready"


    if cache_path.exists() and ready_path.exists():
        return load_from_disk(str(cache_path))


    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))


    if world_size > 1 and rank != 0:
        print(f"[{time.strftime('%H:%M:%S')}] rank={rank} waiting for rank0 to build {cache_name}", flush=True)
        wait_for_cache_ready(cache_path, ready_path)
        return load_from_disk(str(cache_path))


    lang_map = {"te": "telugu", "ta": "tamil", "kn": "kannada", "ml": "malayalam"}
    lang_full = lang_map[language]


    if split == "train":
        src = ROOT / "clean_data" / lang_full / "train" / f"{lang_full}_train_balanced.txt"
    elif split == "val":
        src = ROOT / "clean_data" / lang_full / "val" / f"{lang_full}_val_cleaned_final.txt"
    else:
        raise ValueError(f"Unsupported split: {split}")


    if not src.exists():
        raise FileNotFoundError(f"Missing cleaned {split} file for {language}: {src}")


    error_path = cache_path.parent / f"{cache_name}.error"
    try:
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
        ds = Dataset.from_dict({
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        })
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(cache_path))
        ready_path.write_text("ok\n", encoding="utf-8")
        return ds
    except Exception as exc:
        error_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        raise




def build_multilingual_dataset(
    languages: list[str], split: str, tokenizer: T5Tokenizer, seed: int
) -> Dataset:
    datasets = []
    for lang in languages:
        rank = int(os.environ.get("RANK", "0"))
        if rank == 0:
            print(f"[{time.strftime('%H:%M:%S')}] loading {lang}/{split}...", flush=True)
        ds = build_or_load_language_dataset(lang, split, tokenizer)
        datasets.append(ds)
        if rank == 0:
            print(f"[{time.strftime('%H:%M:%S')}]   {lang}: {len(ds):,} examples", flush=True)


    combined = concatenate_datasets(datasets)
    combined = combined.shuffle(seed=seed)
    return combined




def compute_total_steps(
    dataset_size: int, batch_size: int, grad_accum: int, num_gpus: int, num_epochs: int
) -> int:
    steps_per_epoch = max(1, dataset_size // (batch_size * grad_accum * max(1, num_gpus)))
    return steps_per_epoch * num_epochs




def train_one(languages: list[str], seed: int, tokenizer_dirname: str) -> None:
    run_name = f"seed{seed}"
    lang_tag = "multilingual"
    output_dir = ROOT / "models" / lang_tag / run_name
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / f"{lang_tag}_{run_name}.json"


    rank = int(os.environ.get("RANK", "0"))


    if output_dir.exists() and log_path.exists():
        if rank == 0:
            print(f"[{time.strftime('%H:%M:%S')}] skipping {lang_tag}/{run_name} — already done", flush=True)
        return


    if rank == 0:
        print(f"[{time.strftime('%H:%M:%S')}] starting {lang_tag}/{run_name} — languages: {languages}", flush=True)
    set_seed(seed)


    tokenizer = load_tokenizer(tokenizer_dirname)


    train_dataset = build_multilingual_dataset(languages, "train", tokenizer, seed)
    val_dataset = build_multilingual_dataset(languages, "val", tokenizer, seed)


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
            "language": lang_tag,
            "languages": languages,
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
            f"[{time.strftime('%H:%M:%S')}] done {lang_tag}/{run_name} — "
            f"eval_loss={eval_loss:.4f}, perplexity={perplexity:.2f}",
            flush=True,
        )




def main() -> None:
    args = parse_args()
    languages = [lang.strip() for lang in args.languages.split(",") if lang.strip()]


    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        print(f"=== MULTILINGUAL ({', '.join(languages)}) ===", flush=True)


    for seed in range(START_SEED, START_SEED + NUM_SEEDS):
        if rank == 0:
            print(f"=== seed={seed} ===", flush=True)
        train_one(languages, seed, args.tokenizer_dirname)


    if rank == 0:
        print("=== MULTILINGUAL DONE ===", flush=True)




if __name__ == "__main__":
    main()




