#!/usr/bin/env python3

from __future__ import annotations

"""Evaluate the pruned-data GPT-2 variants (and the 100% baseline) on a
common footing: held-out BPB/perplexity plus WikiANN NER.

Step 4 of the data-pruning scaling-law experiment. For each variant this
writes results/raw/{language}_pruning_{variant}.json combining:
  - perplexity/BPB on the held-out test split (evaluation.perplexity)
  - WikiANN NER F1 (evaluation.downstream)
  - the training curve (`eval_history`) copied in from the matching
    training-run JSON, when one exists

The 100% baseline ("full") is loaded from the HF Hub id used elsewhere in
this repo (`evaluation.run_eval.MODEL_ID`) rather than a local checkpoint,
per the existing telugu_seed2 run — its training curve stays the coarse,
already-recorded one rather than being re-measured here.

Usage
-----
python -m dravidian_lm.pruning.evaluate_variants --language telugu --language_code te --tokenizer_name te
"""

import argparse
import gc
import json
import os

import torch
from transformers import GPT2LMHeadModel

from dravidian_lm.paths import MODELS_DIR, RAW_RESULTS_DIR
from dravidian_lm.evaluation.perplexity import run_perplexity_suite
from dravidian_lm.evaluation.downstream import run_wikiann_ner
from dravidian_lm.evaluation.run_eval import MODEL_ID, load_model_and_tokenizer
from dravidian_lm.models.gpt2.train import load_tokenizer as load_our_tokenizer


ALL_VARIANTS = ("easy", "hard", "mid", "random", "full")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate pruning-experiment GPT-2 checkpoints for BPB and WikiANN NER."
    )
    parser.add_argument("--language", default="telugu")
    parser.add_argument("--language_code", default="te")
    parser.add_argument("--tokenizer_name", default="te")
    parser.add_argument("--variants", nargs="+", choices=ALL_VARIANTS, default=list(ALL_VARIANTS))
    parser.add_argument("--seed", type=int, default=1, help="Seed used when training the pruned variants.")
    parser.add_argument(
        "--baseline_run_name",
        default="seed2",
        help="results/raw/{language}_{baseline_run_name}.json — the existing 100% training run to echo.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_eval_lines", type=int, default=5_000)
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()


def load_variant(variant: str, args: argparse.Namespace, device: str):
    """Returns (model, tokenizer, checkpoint_ref)."""
    if variant == "full":
        model, tokenizer = load_model_and_tokenizer(MODEL_ID, device)
        return model, tokenizer, MODEL_ID

    checkpoint_dir = MODELS_DIR / "gpt2" / args.language / f"seed{args.seed}_{variant}"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"No checkpoint for variant={variant} at {checkpoint_dir}")

    tokenizer = load_our_tokenizer(args.tokenizer_name)
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = GPT2LMHeadModel.from_pretrained(str(checkpoint_dir), dtype=dtype)
    model.to(device)
    model.eval()
    return model, tokenizer, str(checkpoint_dir)


def load_training_eval_history(language: str, run_name: str) -> list[dict] | None:
    path = RAW_RESULTS_DIR / f"{language}_{run_name}.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("eval_history")


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    RAW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for variant in args.variants:
        print(f"\n{'=' * 62}\n  VARIANT: {variant}\n{'=' * 62}")
        model, tokenizer, checkpoint_ref = load_variant(variant, args, device)

        output: dict = {
            "language": args.language,
            "language_code": args.language_code,
            "variant": variant,
            "checkpoint": checkpoint_ref,
        }

        print("-- perplexity / BPB --")
        output["perplexity"] = run_perplexity_suite(
            tokenizer=tokenizer,
            model=model,
            language_code=args.language_code,
            device=device,
            max_eval_lines=args.max_eval_lines,
            batch_size=args.batch_size,
        )

        print("-- WikiANN NER --")
        try:
            ner_result = run_wikiann_ner(checkpoint_ref, tokenizer, output_dir="/tmp/dravidian_pruning_ner")
            output["wikiann_ner"] = ner_result.to_dict()
        except Exception as exc:
            print(f"  [wikiann_ner] FAILED: {exc}")
            output["wikiann_ner"] = {"error": str(exc)}

        run_name = args.baseline_run_name if variant == "full" else f"seed{args.seed}_{variant}"
        eval_history = load_training_eval_history(args.language, run_name)
        if eval_history is not None:
            output["eval_history"] = eval_history
        else:
            print(f"  [eval_history] no training curve found for {args.language}_{run_name}.json")

        out_path = RAW_RESULTS_DIR / f"{args.language}_pruning_{variant}.json"
        out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved -> {out_path}")

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nAll variants evaluated.")


if __name__ == "__main__":
    main()
