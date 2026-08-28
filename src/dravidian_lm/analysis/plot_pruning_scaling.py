#!/usr/bin/env python3

from __future__ import annotations

"""Plot Tokens Trained vs Validation BPB for the data-pruning scaling-law experiment.

Reads each variant's training curve (`eval_history`, written by
models.gpt2.train and echoed into results/raw/{language}_pruning_{variant}.json
by pruning.evaluate_variants) and plots tokens_seen vs bpb. The 100% baseline
has no fine-grained curve (by design — see docs/pruning_experiment.md), so it
is drawn as a single point / dashed reference line at its final BPB instead.

Usage
-----
python -m dravidian_lm.analysis.plot_pruning_scaling --language telugu
"""

import argparse
import json

import matplotlib.pyplot as plt

from dravidian_lm.paths import RAW_RESULTS_DIR, RESULTS_DIR


PLOTS_DIR = RESULTS_DIR / "plots"

VARIANT_STYLE = {
    "full": {"color": "black", "label": "100% baseline"},
    "mid": {"color": "#1f77b4", "label": "D_mid (Pareto core-set, 50%)"},
    "random": {"color": "#7f7f7f", "label": "D_random (50%)"},
    "easy": {"color": "#2ca02c", "label": "D_easy (bottom 50% loss)"},
    "hard": {"color": "#d62728", "label": "D_hard (top 50% loss)"},
}

# Plot order matters for legend clarity: primary deliverable comparison first.
VARIANT_ORDER = ["full", "mid", "random", "easy", "hard"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot tokens-trained vs validation BPB across pruning variants.")
    parser.add_argument("--language", default="telugu")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=VARIANT_ORDER,
        choices=VARIANT_ORDER,
    )
    parser.add_argument("--out", default=None, help="Override output PNG path.")
    return parser.parse_args()


def load_variant_result(language: str, variant: str) -> dict | None:
    path = RAW_RESULTS_DIR / f"{language}_pruning_{variant}.json"
    if not path.exists():
        print(f"  [{variant}] no results file at {path}, skipping")
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    plotted_any = False

    for variant in args.variants:
        result = load_variant_result(args.language, variant)
        if result is None:
            continue
        style = VARIANT_STYLE[variant]
        history = result.get("eval_history") or []

        if variant == "full":
            overall = result.get("perplexity", {}).get("overall", {})
            final_bpb = overall.get("bpb")
            if final_bpb is None and history:
                final_bpb = history[-1]["bpb"]
            if final_bpb is not None:
                ax.axhline(
                    final_bpb, color=style["color"], linestyle="--", linewidth=1.5, label=style["label"]
                )
                plotted_any = True
            else:
                print("  [full] no BPB available to plot (run pruning.evaluate_variants first)")
            continue

        points = [(h["tokens_seen"], h["bpb"]) for h in history if h.get("bpb") is not None]
        if not points:
            print(f"  [{variant}] no eval_history points to plot")
            continue
        points.sort()
        xs, ys = zip(*points)
        ax.plot(xs, ys, marker="o", color=style["color"], label=style["label"])
        plotted_any = True

    if not plotted_any:
        raise RuntimeError(
            "Nothing to plot — run pruning.evaluate_variants (and check for eval_history) first."
        )

    ax.set_xlabel("Tokens trained")
    ax.set_ylabel("Validation BPB (bits per byte)")
    ax.set_title(f"{args.language.capitalize()}: data pruning vs scaling (BPB)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = args.out or (PLOTS_DIR / f"{args.language}_pruning_scaling_bpb.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
