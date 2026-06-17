from __future__ import annotations

import json

from dravidian_lm.paths import RAW_RESULTS_DIR


def load_rows() -> list[dict]:
    rows: list[dict] = []
    for path in sorted(RAW_RESULTS_DIR.glob("*.json")):
        with path.open(encoding="utf-8") as f:
            payload = json.load(f)
        rows.append(
            {
                "language": payload.get("language", "unknown"),
                "architecture": payload.get("architecture", "gpt2"),
                "run_name": payload.get("run_name", path.stem),
                "eval_loss": payload.get("eval_loss"),
                "perplexity": payload.get("perplexity"),
                "path": path,
            }
        )
    return rows


def format_metric(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def main() -> None:
    rows = load_rows()
    if not rows:
        print(f"No result files found in {RAW_RESULTS_DIR}")
        return

    print("| Language | Architecture | Run | Eval Loss | Perplexity |")
    print("| --- | --- | --- | ---: | ---: |")
    for row in rows:
        print(
            f"| {row['language']} | {row['architecture']} | {row['run_name']} | "
            f"{format_metric(row['eval_loss'])} | {format_metric(row['perplexity'])} |"
        )


if __name__ == "__main__":
    main()
