from __future__ import annotations

"""Main evaluation entrypoint for Dravidian LM research.

Usage examples
--------------
# Full evaluation (GPU recommended):
python -m dravidian_lm.evaluation.run_eval --tasks all

# Intrinsic only (fast, CPU-friendly):
python -m dravidian_lm.evaluation.run_eval --tasks perplexity tokenizer

# Downstream + mGPT baseline:
python -m dravidian_lm.evaluation.run_eval --tasks downstream --run_baselines

# Specific downstream tasks only:
python -m dravidian_lm.evaluation.run_eval --tasks downstream --downstream_tasks indicsentiment wikiann_ner

Outputs
-------
results/raw/{language}_full_eval.json — all metrics in a single JSON file
"""

import argparse
import json
import re
import time
from datetime import datetime, timezone

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, T5TokenizerFast, set_seed

from dravidian_lm.paths import RAW_RESULTS_DIR
from dravidian_lm.evaluation.perplexity import download_split, load_texts, run_perplexity_suite
from dravidian_lm.evaluation.tokenizer_analysis import run_tokenizer_comparison
from dravidian_lm.evaluation.downstream import run_downstream_suite


MODEL_ID = "pulipakav-1/dravidian-gpt2-telugu"
LANGUAGE = "telugu"
LANGUAGE_CODE = "te"

TELUGU_ONLY_RE = re.compile(r"[^ఀ-౿\s.,;:!?()\-\"'0-9]")

GENERATION_PROMPTS = [
    "తెలుగు భాష గురించి ఒక చిన్న పేరా రాయండి.",
    "విజయవాడ నగరంపై ఐదు వాక్యాలు రాయండి.",
    "వర్షపు రోజు గురించి ఒక చిన్న కథ మొదలు పెట్టండి.",
    "భవిష్యత్తులో విద్య ఎలా మారుతుంది అనే విషయంపై తెలుగులో రాయండి.",
    "కృషి మరియు రైతుల ప్రాముఖ్యతపై ఒక చిన్న వ్యాఖ్యానం రాయండి.",
    "తెలుగు సాహిత్యం",
    "భారతదేశంలో అనేక",
    "ప్రకృతి అందాలు",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Dravidian GPT-2 models for research.")
    parser.add_argument(
        "--model_name",
        default=MODEL_ID,
        help="HuggingFace model ID or local path.",
    )
    parser.add_argument(
        "--language",
        default=LANGUAGE,
        help="Full language name (used in output filenames).",
    )
    parser.add_argument(
        "--language_code",
        default=LANGUAGE_CODE,
        help="Two-letter language code matching data/splits/ directory.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["all", "perplexity", "tokenizer", "downstream", "generation"],
        default=["all"],
        help="Which evaluation tasks to run.",
    )
    parser.add_argument(
        "--downstream_tasks",
        nargs="+",
        choices=["indicsentiment", "wikiann_ner", "indicxnli"],
        default=None,
        help="Downstream tasks to run (default: all three).",
    )
    parser.add_argument(
        "--run_baselines",
        action="store_true",
        help="Re-run downstream tasks with mGPT for direct comparison.",
    )
    parser.add_argument(
        "--max_eval_lines",
        type=int,
        default=5_000,
        help="Maximum lines to load from the test split for perplexity.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for perplexity evaluation.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override (e.g. 'cpu', 'cuda', 'cuda:1'). Auto-detects if omitted.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return parser.parse_args()


def load_tokenizer_from_hub(model_name: str) -> T5TokenizerFast:
    """Load the tokenizer directly from the raw SentencePiece binary.

    Several of our Hub repos (telugu, kannada, malayalam) carry a stale
    tokenizer.json with a single-token stub vocab left over from an upload
    step — loading via AutoTokenizer/PreTrainedTokenizerFast silently maps
    every input to <unk>. Every repo also carries the real tokenizer.model
    SentencePiece binary (the same file train.py's tokenizer was built
    from), so always load from that instead of trusting tokenizer.json.

    Must use T5TokenizerFast, not the slow T5Tokenizer: the slow tokenizer
    misparses this custom Unigram SentencePiece model (silently collapses
    to a ~4-token vocab) — see train.py's load_tokenizer() and the "Use
    T5TokenizerFast instead of slow Python T5Tokenizer" fix in this repo's
    history.
    """
    vocab_file = hf_hub_download(model_name, "tokenizer.model")
    tokenizer = T5TokenizerFast(vocab_file=vocab_file, extra_ids=0)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    return tokenizer


def load_model_and_tokenizer(model_name: str, device: str):
    print(f"Loading model: {model_name}")
    tokenizer = load_tokenizer_from_hub(model_name)

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    model.to(device)
    model.eval()

    # Override generation token IDs to match the custom tokeniser
    for cfg in (model.config, model.generation_config):
        cfg.pad_token_id = tokenizer.pad_token_id
        cfg.eos_token_id = tokenizer.eos_token_id
        cfg.bos_token_id = tokenizer.eos_token_id  # no dedicated BOS in this tokeniser

    print(f"  vocab_size={tokenizer.vocab_size}, device={device}, dtype={dtype}")
    return model, tokenizer


def run_generation(model, tokenizer, device: str) -> list[dict]:
    results = []
    for i, prompt in enumerate(GENERATION_PROMPTS, start=1):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        continuation = raw[len(prompt):].lstrip() if raw.startswith(prompt) else raw
        # Keep only Telugu script for display
        telugu_only = re.sub(r"\s+", " ", TELUGU_ONLY_RE.sub(" ", continuation.replace("▁", " "))).strip()
        results.append({"id": i, "prompt": prompt, "raw_continuation": continuation, "telugu_only": telugu_only})
        print(f"  [generation] prompt {i}: {telugu_only[:80]} ...")
    return results


def resolve_tasks(task_flags: list[str]) -> set[str]:
    if "all" in task_flags:
        return {"perplexity", "tokenizer", "downstream", "generation"}
    return set(task_flags)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tasks = resolve_tasks(args.tasks)

    model, tokenizer = load_model_and_tokenizer(args.model_name, device)

    output: dict = {
        "model": args.model_name,
        "language": args.language,
        "language_code": args.language_code,
        "evaluation_date": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "seed": args.seed,
    }

    # ---- Perplexity + BPB ----
    if "perplexity" in tasks:
        print("\n=== Perplexity ===")
        t0 = time.time()
        output["perplexity"] = run_perplexity_suite(
            tokenizer=tokenizer,
            model=model,
            language=args.language,
            device=device,
            max_eval_lines=args.max_eval_lines,
            batch_size=args.batch_size,
        )
        print(f"  done in {time.time() - t0:.0f}s")

    # ---- Tokeniser analysis ----
    if "tokenizer" in tasks:
        print("\n=== Tokeniser Analysis ===")
        t0 = time.time()
        test_path = download_split(args.language, "test")
        sample_texts = load_texts(test_path, max_lines=2_000)
        output["tokenizer_analysis"] = run_tokenizer_comparison(
            texts=sample_texts,
            our_tokenizer=tokenizer,
            our_name=args.model_name.split("/")[-1],
        )
        print(f"  done in {time.time() - t0:.0f}s")

    # ---- Downstream fine-tuning ----
    if "downstream" in tasks:
        print("\n=== Downstream Tasks ===")
        t0 = time.time()
        output["downstream"] = run_downstream_suite(
            model_name=args.model_name,
            our_tokenizer=tokenizer,
            device=device,
            tasks=args.downstream_tasks,
            run_baselines=args.run_baselines,
            language_code=args.language_code,
        )
        print(f"  done in {time.time() - t0:.0f}s")

    # ---- Generation samples ----
    if "generation" in tasks:
        print("\n=== Generation Samples ===")
        t0 = time.time()
        output["generation_samples"] = run_generation(model, tokenizer, device)
        print(f"  done in {time.time() - t0:.0f}s")

    # ---- Save ----
    RAW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_RESULTS_DIR / f"{args.language}_full_eval.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved → {out_path}")

    # ---- Console summary ----
    print("\n========== SUMMARY ==========")
    if "perplexity" in output:
        overall = output["perplexity"].get("overall", {})
        if overall:
            print(
                f"Perplexity (overall): "
                f"loss={overall['eval_loss']:.4f}  "
                f"ppl={overall['perplexity']:.2f}  "
                f"bpb={overall['bpb']:.4f}"
            )
    if "tokenizer_analysis" in output:
        our_key = args.model_name.split("/")[-1]
        our = output["tokenizer_analysis"].get(our_key, {})
        if our:
            print(
                f"Tokeniser ({our_key}): "
                f"fertility={our['fertility']:.2f}  "
                f"compression={our['compression_ratio']:.2f} bytes/tok"
            )
    if "downstream" in output:
        for task, models in output["downstream"].items():
            for mdl, res in models.items():
                if isinstance(res, dict) and "score" in res:
                    print(f"Downstream {task} ({mdl}): {res['metric_name']}={res['score']:.4f}")
    print("=============================")


if __name__ == "__main__":
    main()
