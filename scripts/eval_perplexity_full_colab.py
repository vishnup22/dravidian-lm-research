from __future__ import annotations

"""Full-test-set perplexity evaluation, meant to be run from a Colab notebook.

Colab setup (run in a cell before this script):
    !git clone https://github.com/<you>/dravidian-lm-research.git
    %cd dravidian-lm-research
    !pip install -e .

Usage
-----
# All four monolingual models, full held-out test split each:
!python scripts/eval_perplexity_full_colab.py

# Just one language, or a smaller line cap for a quick smoke test:
!python scripts/eval_perplexity_full_colab.py --languages tamil --max_eval_lines 2000

# Also score the multilingual model against every language's test split:
!python scripts/eval_perplexity_full_colab.py --include_multi

Each language is evaluated on its FULL test split by default (no line cap),
which for tamil (~300MB of text) can take a couple of hours on a single
Colab GPU. Results are written incrementally to results/raw/ after each
model finishes, so a session timeout only loses the in-flight model.
"""

import argparse
import json
import time
from datetime import datetime, timezone

import torch

from dravidian_lm.evaluation.perplexity import run_perplexity_suite
from dravidian_lm.evaluation.run_eval import load_model_and_tokenizer
from dravidian_lm.paths import RAW_RESULTS_DIR

LANGUAGE_TO_MODEL = {
    "telugu": "pulipakav-1/dravidian-gpt2-telugu",
    "tamil": "pulipakav-1/dravidian-gpt2-tamil",
    "kannada": "pulipakav-1/dravidian-gpt2-kannada",
    "malayalam": "pulipakav-1/dravidian-gpt2-malayalam",
}
MULTI_MODEL_ID = "pulipakav-1/dravidian-gpt2-multi"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-test-set perplexity eval for Colab.")
    parser.add_argument(
        "--languages",
        nargs="+",
        choices=list(LANGUAGE_TO_MODEL),
        default=list(LANGUAGE_TO_MODEL),
        help="Which monolingual models to evaluate (default: all four).",
    )
    parser.add_argument(
        "--include_multi",
        action="store_true",
        help="Also evaluate dravidian-gpt2-multi against every selected language's test split.",
    )
    parser.add_argument(
        "--skip_mono",
        action="store_true",
        help="Skip the monolingual models entirely (use with --include_multi to only score multi).",
    )
    parser.add_argument(
        "--max_eval_lines",
        type=int,
        default=None,
        help="Cap on test lines per language. Omit for the FULL test split.",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def eval_one(model_id: str, language: str, device: str, max_eval_lines, batch_size: int, tag: str) -> dict:
    print(f"\n=== {tag}: {model_id} on {language} test (full) ===")
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(model_id, device)
    result = run_perplexity_suite(
        tokenizer=tokenizer,
        model=model,
        language=language,
        device=device,
        max_eval_lines=max_eval_lines,
        batch_size=batch_size,
    )
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.0f}s")

    record = {
        "model": model_id,
        "language": language,
        "evaluation_date": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "max_eval_lines": max_eval_lines,
        "elapsed_seconds": round(elapsed, 1),
        "perplexity": result,
    }
    RAW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_RESULTS_DIR / f"{tag}_{language}_perplexity_full.json"
    out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  saved -> {out_path}")
    return record


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  languages={args.languages}  max_eval_lines={args.max_eval_lines or 'FULL'}")

    all_results = []

    if not args.skip_mono:
        for language in args.languages:
            model_id = LANGUAGE_TO_MODEL[language]
            all_results.append(
                eval_one(model_id, language, device, args.max_eval_lines, args.batch_size, tag="mono")
            )

    if args.include_multi:
        for language in args.languages:
            all_results.append(
                eval_one(MULTI_MODEL_ID, language, device, args.max_eval_lines, args.batch_size, tag="multi")
            )

    print("\n========== SUMMARY ==========")
    for r in all_results:
        overall = r["perplexity"]["overall"]
        print(
            f"[{r['model'].split('/')[-1]:>28}] {r['language']:<10} "
            f"loss={overall['eval_loss']:.4f}  ppl={overall['perplexity']:.2f}  "
            f"bpb={overall['bpb']:.4f}  tokens={overall['num_tokens']:,}"
        )
    print("=============================")


if __name__ == "__main__":
    main()
