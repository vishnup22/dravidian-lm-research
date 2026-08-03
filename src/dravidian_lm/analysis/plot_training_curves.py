from __future__ import annotations

"""Plot pretraining loss/perplexity curves from each model's final HF Hub checkpoint.

Trainer's log_history is cumulative — the final checkpoint's trainer_state.json
already contains the full training-loss (every logging_steps) and eval-loss
(every eval epoch) history for the whole run, so we only need one small JSON
download per model, never the multi-hundred-MB model/optimizer weights.

Kannada has no checkpoint subfolders on the Hub (only the final merged model
was uploaded), so it has no curve here — see results/raw/kannada_seed1.json
for its single final data point instead.
"""

import json
import math

import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

from dravidian_lm.paths import RESULTS_DIR


FINAL_CHECKPOINTS = {
    "telugu": ("pulipakav-1/dravidian-gpt2-telugu", "checkpoint-268446"),
    "tamil": ("pulipakav-1/dravidian-gpt2-tamil", "checkpoint-540278"),
    "malayalam": ("pulipakav-1/dravidian-gpt2-malayalam", "checkpoint-340491"),
    "multi": ("pulipakav-1/dravidian-gpt2-multi", "checkpoint-1078248"),
}

FIGURES_DIR = RESULTS_DIR / "figures"


def load_log_history(repo_id: str, checkpoint: str) -> dict:
    path = hf_hub_download(repo_id, f"{checkpoint}/trainer_state.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def split_history(log_history: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split log_history into (training-loss entries, eval entries)."""
    train_entries = [e for e in log_history if "loss" in e]
    eval_entries = [e for e in log_history if "eval_loss" in e]
    return train_entries, eval_entries


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    curves: dict = {}

    for name, (repo_id, checkpoint) in FINAL_CHECKPOINTS.items():
        print(f"[{name}] downloading {checkpoint}/trainer_state.json from {repo_id} ...")
        state = load_log_history(repo_id, checkpoint)
        train_entries, eval_entries = split_history(state["log_history"])
        curves[name] = {
            "train_steps": [e["step"] for e in train_entries],
            "train_loss": [e["loss"] for e in train_entries],
            "grad_norm": [e.get("grad_norm") for e in train_entries],
            "learning_rate": [e.get("learning_rate") for e in train_entries],
            "eval_epoch": [e["epoch"] for e in eval_entries],
            "eval_loss": [e["eval_loss"] for e in eval_entries],
            "eval_perplexity": [math.exp(min(e["eval_loss"], 20.0)) for e in eval_entries],
        }
        print(
            f"  {len(train_entries)} training-loss points, "
            f"{len(eval_entries)} eval points, "
            f"final eval_loss={eval_entries[-1]['eval_loss']:.4f}"
        )

    out_path = RESULTS_DIR / "raw" / "training_curves.json"
    out_path.write_text(json.dumps(curves, indent=2), encoding="utf-8")
    print(f"Saved curve data -> {out_path}")

    # ---- Figure 1: training loss vs. step ----
    plt.figure(figsize=(7, 4.5))
    for name, c in curves.items():
        plt.plot(c["train_steps"], c["train_loss"], label=name, linewidth=1.2)
    plt.xlabel("Training step")
    plt.ylabel("Training loss (cross-entropy, nats)")
    plt.title("Pretraining loss")
    plt.legend()
    plt.tight_layout()
    loss_path = FIGURES_DIR / "training_loss.png"
    plt.savefig(loss_path, dpi=200)
    plt.savefig(FIGURES_DIR / "training_loss.pdf")
    plt.close()
    print(f"Saved -> {loss_path}")

    # ---- Per-model figures: training loss only ----
    for name, c in curves.items():
        plt.figure(figsize=(7, 4.5))
        plt.plot(c["train_steps"], c["train_loss"], linewidth=1.0, color="tab:blue")
        plt.xlabel("Training step")
        plt.ylabel("Training loss (cross-entropy, nats)")
        plt.title(f"dravidian-gpt2-{name}: training loss")
        plt.tight_layout()
        model_path = FIGURES_DIR / f"{name}_training_curve.png"
        plt.savefig(model_path, dpi=200)
        plt.savefig(FIGURES_DIR / f"{name}_training_curve.pdf")
        plt.close()
        print(f"Saved -> {model_path}")


if __name__ == "__main__":
    main()
