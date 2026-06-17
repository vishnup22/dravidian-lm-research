from __future__ import annotations

"""Downstream fine-tuning evaluation for Telugu GPT-2.

Tasks:
  indicsentiment  — binary sentiment classification (ai4bharat/IndicSentiment, te)
  wikiann_ner     — named entity recognition (wikiann, te)
  indicxnli       — natural language inference (ai4bharat/IndicXNLI or xnli, te)

For each task we fine-tune a linear head on top of the frozen or fully
unfrozen backbone using HuggingFace Trainer, then report test-set metrics.
Passing --run_baselines will repeat the same protocol for mGPT so numbers
are directly comparable.
"""

import os
import sys
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch

# torchvision >= 0.17 removed VideoReader; stub it so Trainer callbacks don't crash.
try:
    import torchvision.io as _tvio
    if not hasattr(_tvio, "VideoReader"):
        class _VideoReaderStub:
            pass
        _tvio.VideoReader = _VideoReaderStub
        sys.modules.setdefault("torchvision.io", _tvio)
except Exception:
    pass
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    DataCollatorWithPadding,
    GPT2Config,
    GPT2ForSequenceClassification,
    GPT2ForTokenClassification,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)


DOWNSTREAM_MAX_LEN = 256
FINETUNE_EPOCHS = 5
FINETUNE_LR = 2e-5
FINETUNE_BATCH = 16
WARMUP_RATIO = 0.1
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
# Shared helpers
# ---------------------------------------------------------------------------

def _trainer_args(output_dir: str, use_bf16: bool) -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=FINETUNE_EPOCHS,
        per_device_train_batch_size=FINETUNE_BATCH,
        per_device_eval_batch_size=FINETUNE_BATCH,
        learning_rate=FINETUNE_LR,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=use_bf16,
        report_to="none",
        logging_steps=50,
    )


def _load_seq_clf_model(
    model_name: str,
    num_labels: int,
    pad_token_id: int,
) -> GPT2ForSequenceClassification:
    config = GPT2Config.from_pretrained(
        model_name,
        num_labels=num_labels,
        pad_token_id=pad_token_id,
    )
    model = GPT2ForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        ignore_mismatched_sizes=True,
    )
    model.config.pad_token_id = pad_token_id
    return model


def _tok_for_model(model_name: str, our_tokenizer: PreTrainedTokenizer) -> PreTrainedTokenizer:
    """Return the appropriate tokeniser for a given model."""
    if model_name == "ai-forever/mGPT":
        tok = AutoTokenizer.from_pretrained(model_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
            tok.pad_token_id = tok.eos_token_id
        return tok
    return our_tokenizer


# ---------------------------------------------------------------------------
# IndicSentiment (binary sentiment)
# ---------------------------------------------------------------------------

def _load_indicsentiment_te() -> DatasetDict:
    """Load IndicSentiment Telugu from raw JSON files (loading script is blocked by HF).

    Repo layout: data/{test,validation}/te.json — no train split exists for Telugu.
    We create a synthetic train split by holding out 80 % of the validation set.
    """
    import json
    from huggingface_hub import hf_hub_download

    splits: dict[str, Dataset] = {}
    for hf_split in ("test", "validation"):
        repo_path = f"data/{hf_split}/te.json"
        try:
            local = hf_hub_download("ai4bharat/IndicSentiment", repo_path, repo_type="dataset")
        except Exception as exc:
            raise RuntimeError(f"Cannot download {repo_path} from ai4bharat/IndicSentiment: {exc}") from exc
        with open(local, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        splits[hf_split] = Dataset.from_list(records)

    # No train split in the repo — carve 80 % of validation for training.
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

    def preprocess(examples: dict) -> dict:
        enc = tokenizer(
            examples[text_col],
            truncation=True,
            max_length=DOWNSTREAM_MAX_LEN,
            padding=False,
        )
        enc["labels"] = examples[label_col]
        return enc

    cols = ds["train"].column_names
    tokenized = ds.map(preprocess, batched=True, remove_columns=cols)
    tokenized.set_format("torch")

    num_labels = len(set(ds["train"][label_col]))
    model = _load_seq_clf_model(model_name, num_labels, tokenizer.pad_token_id)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": float((preds == labels).mean())}

    val_split = "validation" if "validation" in tokenized else "test"
    args = _trainer_args(output_dir, use_bf16=torch.cuda.is_available())
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized[val_split],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    test_metrics = trainer.evaluate(tokenized["test"])

    score = test_metrics.get("eval_accuracy", 0.0)
    return DownstreamResult(
        task="indicsentiment_te",
        model_name=model_name,
        metric_name="accuracy",
        score=round(score, 4),
        num_train=len(tokenized["train"]),
        num_test=len(tokenized["test"]),
        details=test_metrics,
    )


# ---------------------------------------------------------------------------
# WikiAnn NER
# ---------------------------------------------------------------------------

def _align_ner_labels(
    words: list[str],
    ner_tags: list[int],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
) -> dict:
    """Tokenise words individually and align NER labels to subword tokens.

    The first subword of each word inherits the word's label; subsequent
    subwords are masked with -100 so they do not contribute to the loss.
    This approach works with both fast and slow tokenisers.
    """
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
    id2label = {i: l for i, l in enumerate(label_list)}
    label2id = {l: i for i, l in enumerate(label_list)}

    def preprocess(example: dict) -> dict:
        return _align_ner_labels(
            example["tokens"], example["ner_tags"], tokenizer, DOWNSTREAM_MAX_LEN
        )

    cols = [c for c in ds["train"].column_names if c not in ("input_ids", "attention_mask", "labels")]
    tokenized = ds.map(preprocess, remove_columns=cols)
    tokenized.set_format("torch")

    config = GPT2Config.from_pretrained(
        model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = GPT2ForTokenClassification.from_pretrained(
        model_name, config=config, ignore_mismatched_sizes=True
    )

    import seqeval.metrics as seqeval_metrics

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        true_preds = [
            [id2label[p] for p, l in zip(pred_row, label_row) if l != -100]
            for pred_row, label_row in zip(preds, labels)
        ]
        true_labels = [
            [id2label[l] for l in label_row if l != -100]
            for label_row in labels
        ]
        return {
            "f1": seqeval_metrics.f1_score(true_labels, true_preds),
            "precision": seqeval_metrics.precision_score(true_labels, true_preds),
            "recall": seqeval_metrics.recall_score(true_labels, true_preds),
        }

    args = _trainer_args(output_dir, use_bf16=torch.cuda.is_available())
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    test_metrics = trainer.evaluate(tokenized["test"])

    score = test_metrics.get("eval_f1", 0.0)
    return DownstreamResult(
        task="wikiann_ner_te",
        model_name=model_name,
        metric_name="f1",
        score=round(score, 4),
        num_train=len(tokenized["train"]),
        num_test=len(tokenized["test"]),
        details=test_metrics,
    )


# ---------------------------------------------------------------------------
# IndicXNLI (natural language inference — cross-lingual transfer)
# ---------------------------------------------------------------------------

def run_indicxnli(
    model_name: str,
    tokenizer: PreTrainedTokenizer,
    output_dir: str,
) -> DownstreamResult:
    """Fine-tune on translated Telugu XNLI training data and evaluate on test.

    Note: cross-lingual transfer from a monolingual Telugu LM to NLI is a
    stringent test; lower scores vs mGPT are expected and informative.
    """
    # Telugu is not in XNLI (facebook/xnli covers 15 languages, not Telugu).
    # ai4bharat/IndicXNLI is also unavailable (deprecated loading script).
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

    return results
