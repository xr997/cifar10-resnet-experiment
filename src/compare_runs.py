#!/usr/bin/env python3
"""Aggregate and plot CIFAR-10 experiment summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_metrics(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append({key: float(value) for key, value in row.items()})
        return rows


def plot_comparison(run_dirs: list[Path], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), dpi=180)
    labels = []
    test_accs = []
    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed"]

    for index, run_dir in enumerate(run_dirs):
        summary = read_json(run_dir / "summary.json")
        metrics = read_metrics(run_dir / "metrics.csv")
        label = summary["model"]
        labels.append(label)
        test_accs.append(summary["test_acc"])
        best_row = max(metrics, key=lambda row: row["val_acc"])
        axes[0].plot(
            [row["epoch"] for row in metrics],
            [row["val_acc"] for row in metrics],
            linewidth=2.0,
            color=colors[index % len(colors)],
            label=label,
        )
        axes[0].scatter(
            [best_row["epoch"]],
            [best_row["val_acc"]],
            s=36,
            color=colors[index % len(colors)],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )

    axes[0].set_title("Validation Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    all_val_accs = [row["val_acc"] for run_dir in run_dirs for row in read_metrics(run_dir / "metrics.csv")]
    axes[0].set_ylim(max(0.0, min(all_val_accs) - 0.08), 1.0)
    max_epoch = max(int(row["epoch"]) for run_dir in run_dirs for row in read_metrics(run_dir / "metrics.csv"))
    tick_step = 10 if max_epoch >= 40 else max(1, max_epoch // 5)
    axes[0].set_xticks([1] + list(range(tick_step, max_epoch + 1, tick_step)))
    axes[0].grid(True, alpha=0.18)
    axes[0].legend(frameon=False)

    bars = axes[1].bar(labels, test_accs, color=colors[: len(labels)], width=0.58)
    min_acc = min(test_accs)
    max_acc = max(test_accs)
    for index, value in enumerate(test_accs):
        axes[1].text(
            index,
            value + 0.002,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[1].set_title("Test Accuracy")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(max(0.0, min_acc - 0.04), min(1.0, max_acc + 0.04))
    axes[1].grid(True, axis="y", alpha=0.18)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    for bar in bars:
        bar.set_linewidth(0)

    fig.tight_layout()
    fig.savefig(output_dir / "ablation_comparison.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, help="Run directories containing summary.json and metrics.csv")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = [Path(item) for item in args.runs]

    summaries = [read_json(run_dir / "summary.json") for run_dir in run_dirs]
    fields = [
        "model",
        "epochs",
        "best_epoch",
        "best_val_acc",
        "test_loss",
        "test_acc",
        "parameter_count",
        "train_size",
        "val_size",
        "test_size",
        "output_dir",
    ]
    with (output_dir / "ablation_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary[field] for field in fields})

    plot_comparison(run_dirs, output_dir)
    print(f"Wrote {output_dir / 'ablation_summary.csv'}")
    print(f"Wrote {output_dir / 'ablation_comparison.png'}")


if __name__ == "__main__":
    main()
