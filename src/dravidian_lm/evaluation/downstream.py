from __future__ import annotations

"""Downstream fine-tuning evaluation for Telugu GPT-2.

Tasks:
  indicsentiment  — binary sentiment classification (ai4bharat/IndicSentiment, te)
  wikiann_ner     — named entity recognition (unimelb-nlp/wikiann, te)
  indicxnli       — skipped (Telugu not in XNLI / loading scripts blocked)

Uses a plain PyTorch training loop instead of HuggingFace Trainer to avoid
callback/dependency issues in Colab environments.
"""

import gc
import os
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    DataCollatorWithPadding,
    GPT2Config,
    GPT2ForSequenceClassification,
    GPT2ForTokenClassification,
    PreTrainedTokenizer,
)


DOWNSTREAM_MAX_LEN = 256
FINETUNE_EPOCHS = 5
FINETUNE_LR = 2e-5
FINETUNE_BATCH = 16
WEIGHT_DECAY = 0.01


@dataclass
class DownstreamResult:
    task: str
    model_name: str
    metric_name: str
    score: float
    num_train: int
    num_test: int
    details: dict

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Plain training loop — no HuggingFace Trainer
# ---------------------------------------------------------------------------

def _train_and_eval(
    model: torch.nn.Module,
    train_ds: Dataset,
    test_ds: Dataset,
    collate_fn,
    device: str,
    epochs: int = FINETUNE_EPOCHS,
    lr: float = FINETUNE_LR,
    batch_size: int = FINETUNE_BATCH,
):
    """Train for `epochs` epochs, return (logits_np, labels_np) on test set."""
    model = model.to(device)
    # fp16 models need float32 optimizer states
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size * 2, collate_fn=collate_fn
    )

    for epoch in range(epochs):
        model.train()
        total_loss, n_steps = 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            n_steps += 1
        print(f"    epoch {epoch + 1}/{epochs}  loss={total_loss / max(n_steps, 1):.4f}")

    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                outputs = model(**batch)
            all_logits.append(outputs.logits.detach().cpu().float().numpy())
            if "labels" in batch:
                all_labels.append(batch["labels"].detach().cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0) if all_labels else np.array([])
    return logits, labels


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _tok_for_model(model_name: str, our_tokenizer: PreTrainedTokenizer) -> PreTrainedTokenizer:
    if model_name == "ai-forever/mGPT":
        tok = AutoTokenizer.from_pretrained(model_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
            tok.pad_token_id = tok.eos_token_id
        return tok
    return our_tokenizer


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_seq_clf_model(
    model_name: str,
    num_labels: int,
    pad_token_id: int,
) -> GPT2ForSequenceClassification:
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    config = GPT2Config.from_pretrained(
        model_name,
        num_labels=num_labels,
        pad_token_id=pad_token_id,
    )
    model = GPT2ForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        ignore_mismatched_sizes=True,
        torch_dtype=dtype,
    )
    model.config.pad_token_id = pad_token_id
    return model


# ---------------------------------------------------------------------------
# IndicSentiment (binary sentiment)
# ---------------------------------------------------------------------------

def _load_indicsentiment_te() -> DatasetDict:
    """Load IndicSentiment Telugu directly from HF Hub file-level download.

    The repo only has data/{test,validation}/te.json (JSONL format).
    We create a synthetic train split from 80 % of validation.
    """
    import json
    from huggingface_hub import hf_hub_download

    splits: dict[str, Dataset] = {}
    for hf_split in ("test", "validation"):
        repo_path = f"data/{hf_split}/te.json"
        try:
            local = hf_hub_download(
                "ai4bharat/IndicSentiment", repo_path, repo_type="dataset"
            )
        except Exception as exc:
            raise RuntimeError(
                f"Cannot download {repo_path} from ai4bharat/IndicSentiment: {exc}"
            ) from exc
        with open(local, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        splits[hf_split] = Dataset.from_list(records)

    val_shuffled = splits["validation"].shuffle(seed=42)
    n_train = int(0.8 * len(val_shuffled))
    splits["train"] = val_shuffled.select(range(n_train))
    splits["validation"] = val_shuffled.select(range(n_train, len(val_shuffled)))
    return DatasetDict(splits)


def run_indicsentiment(
    model_name: str,
    tokenizer: PreTrainedTokenizer,
    output_dir: str,
) -> DownstreamResult:
    print(f"  [indicsentiment] loading ai4bharat/IndicSentiment (te) for {model_name} ...")
    ds: DatasetDict = _load_indicsentiment_te()

    text_col = "INDIC REVIEW"
    label_col = "LABEL"

    def _flat(v):
        return int(v[0]) if isinstance(v, list) else int(v)

    def preprocess(examples: dict) -> dict:
        enc = tokenizer(
            examples[text_col],
            truncation=True,
            max_length=DOWNSTREAM_MAX_LEN,
            padding=False,
        )
        enc["labels"] = [_flat(r) for r in examples[label_col]]
        return enc

    cols = ds["train"].column_names
    tokenized = ds.map(preprocess, batched=True, remove_columns=cols)
    tokenized.set_format("torch")

    flat_train_labels = [_flat(r) for r in ds["train"][label_col]]
    num_labels = len(set(flat_train_labels))
    print(f"    num_labels={num_labels}  train={len(tokenized['train'])}  test={len(tokenized['test'])}")

    model = _load_seq_clf_model(model_name, num_labels, tokenizer.pad_token_id)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    device = _device()

    logits, labels = _train_and_eval(
        model, tokenized["train"], tokenized["test"], collator, device
    )

    preds = np.argmax(logits, axis=-1)
    score = float((preds == labels).mean())

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return DownstreamResult(
        task="indicsentiment_te",
        model_name=model_name,
        metric_name="accuracy",
        score=round(score, 4),
        num_train=len(tokenized["train"]),
        num_test=len(tokenized["test"]),
        details={"accuracy": score},
    )


# ---------------------------------------------------------------------------
# WikiANN NER
# ---------------------------------------------------------------------------

def _align_ner_labels(
    words: list[str],
    ner_tags: list[int],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
) -> dict:
    input_ids: list[int] = []
    labels: list[int] = []

    for word, tag in zip(words, ner_tags):
        word_ids = tokenizer.encode(word, add_special_tokens=False)
        if not word_ids:
            word_ids = [tokenizer.unk_token_id or 1]
        input_ids.extend(word_ids)
        labels.extend([tag] + [-100] * (len(word_ids) - 1))

    input_ids = input_ids[:max_length]
    labels = labels[:max_length]
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def run_wikiann_ner(
    model_name: str,
    tokenizer: PreTrainedTokenizer,
    output_dir: str,
) -> DownstreamResult:
    print(f"  [wikiann_ner] loading unimelb-nlp/wikiann (te) for {model_name} ...")
    ds: DatasetDict = load_dataset("unimelb-nlp/wikiann", "te")

    label_list: list[str] = ds["train"].features["ner_tags"].feature.names
    id2label = {i: lbl for i, lbl in enumerate(label_list)}
    label2id = {lbl: i for i, lbl in enumerate(label_list)}

    def preprocess(example: dict) -> dict:
        return _align_ner_labels(
            example["tokens"], example["ner_tags"], tokenizer, DOWNSTREAM_MAX_LEN
        )

    drop_cols = [
        c for c in ds["train"].column_names
        if c not in ("input_ids", "attention_mask", "labels")
    ]
    tokenized = ds.map(preprocess, remove_columns=drop_cols)
    tokenized.set_format("torch")

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    config = GPT2Config.from_pretrained(
        model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = GPT2ForTokenClassification.from_pretrained(
        model_name, config=config, ignore_mismatched_sizes=True, torch_dtype=dtype
    )

    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    device = _device()

    logits, labels = _train_and_eval(
        model, tokenized["train"], tokenized["test"], collator, device
    )

    import seqeval.metrics as seqeval_metrics

    preds_2d = np.argmax(logits, axis=-1)
    true_preds = [
        [id2label[p] for p, l in zip(pred_row, label_row) if l != -100]
        for pred_row, label_row in zip(preds_2d, labels)
    ]
    true_labels = [
        [id2label[int(l)] for l in label_row if l != -100]
        for label_row in labels
    ]

    f1 = seqeval_metrics.f1_score(true_labels, true_preds)
    precision = seqeval_metrics.precision_score(true_labels, true_preds)
    recall = seqeval_metrics.recall_score(true_labels, true_preds)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return DownstreamResult(
        task="wikiann_ner_te",
        model_name=model_name,
        metric_name="f1",
        score=round(f1, 4),
        num_train=len(tokenized["train"]),
        num_test=len(tokenized["test"]),
        details={"f1": f1, "precision": precision, "recall": recall},
    )


# ---------------------------------------------------------------------------
# IndicXNLI — skipped
# ---------------------------------------------------------------------------

def run_indicxnli(
    model_name: str,
    tokenizer: PreTrainedTokenizer,
    output_dir: str,
) -> DownstreamResult:
    raise RuntimeError(
        "Telugu NLI skipped: ai4bharat/IndicXNLI uses a deprecated loading script "
        "and Telugu is not included in facebook/xnli."
    )


# ---------------------------------------------------------------------------
# Suite runner
# ---------------------------------------------------------------------------

TASK_RUNNERS = {
    "indicsentiment": run_indicsentiment,
    "wikiann_ner": run_wikiann_ner,
    "indicxnli": run_indicxnli,
}


def run_downstream_suite(
    model_name: str,
    our_tokenizer: PreTrainedTokenizer,
    device: str,
    tasks: list[str] | None = None,
    run_baselines: bool = False,
    tmp_dir: str = "/tmp/dravidian_downstream",
) -> dict:
    if tasks is None:
        tasks = list(TASK_RUNNERS.keys())

    models_to_eval: list[str] = [model_name]
    if run_baselines:
        models_to_eval.append("ai-forever/mGPT")

    results: dict = {}

    for task_name in tasks:
        runner = TASK_RUNNERS.get(task_name)
        if runner is None:
            print(f"  [downstream] unknown task: {task_name}")
            continue
        results[task_name] = {}

        for model in models_to_eval:
            tok = _tok_for_model(model, our_tokenizer)
            task_dir = os.path.join(tmp_dir, task_name, model.replace("/", "_"))
            os.makedirs(task_dir, exist_ok=True)
            try:
                result = runner(model, tok, task_dir)
                results[task_name][model] = result.to_dict()
                print(
                    f"  [downstream] {task_name} | {model}: "
                    f"{result.metric_name}={result.score:.4f}"
                )
            except Exception as exc:
                print(f"  [downstream] {task_name} | {model}: FAILED — {exc}")
                results[task_name][model] = {"error": str(exc)}
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    return results
