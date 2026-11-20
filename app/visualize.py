
"""Professional research visualizations for model comparison."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def create_visualizations(
    results_file="results/evaluation_results.json",
    history_file="models/basket_transformer/training_history.json",
    output_dir="results/figures",
):
    """Generate publication-style model comparison figures."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(results_file, "r", encoding="utf-8") as file:
        results = json.load(file)

    with open(history_file, "r", encoding="utf-8") as file:
        history = json.load(file)

    models = list(results.keys())

    # Consistent research-paper styling.
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # -------------------------------------------------
    # FIGURE 1: Top-K Accuracy
    # -------------------------------------------------

    metrics = ["Accuracy@1", "Accuracy@5", "Accuracy@10"]

    x = np.arange(len(metrics))
    width = 0.34

    fig, ax = plt.subplots(figsize=(10, 6))

    for index, model in enumerate(models):
        values = [
            results[model][metric] * 100
            for metric in metrics
        ]

        positions = x + (index - 0.5) * width

        bars = ax.bar(
            positions,
            values,
            width,
            label=model,
        )

        ax.bar_label(
            bars,
            fmt="%.2f%%",
            padding=4,
            fontsize=9,
        )

    ax.set_title(
        "Top-K Prediction Accuracy",
        fontweight="bold",
        pad=18,
    )
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Evaluation Metric")
    ax.set_xticks(x)
    ax.set_xticklabels(["Top-1", "Top-5", "Top-10"])
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(output_path / "01_top_k_accuracy.png")
    plt.close(fig)

    # -------------------------------------------------
    # FIGURE 2: Ranking Performance
    # -------------------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ranking_metrics = [
        ("MRR", "Mean Reciprocal Rank", False),
        ("Mean Rank", "Mean Rank (lower is better)", True),
    ]

    for ax, (metric, title, lower_better) in zip(
        axes, ranking_metrics
    ):
        values = [
            results[model][metric]
            for model in models
        ]

        bars = ax.bar(models, values)

        ax.bar_label(
            bars,
            fmt="%.4f",
            padding=4,
        )

        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    fig.suptitle(
        "Product Ranking Performance",
        fontsize=16,
        fontweight="bold",
    )

    fig.tight_layout()
    fig.savefig(output_path / "02_ranking_performance.png")
    plt.close(fig)

    # -------------------------------------------------
    # FIGURE 3: Training and Validation Loss
    # -------------------------------------------------

    epochs = [record["epoch"] for record in history]
    train_loss = [record["train_loss"] for record in history]
    val_loss = [record["val_loss"] for record in history]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        epochs,
        train_loss,
        marker="o",
        linewidth=2.5,
        label="Training Loss",
    )

    ax.plot(
        epochs,
        val_loss,
        marker="s",
        linewidth=2.5,
        label="Validation Loss",
    )

    best_epoch = int(np.argmin(val_loss)) + 1

    ax.axvline(
        best_epoch,
        linestyle="--",
        alpha=0.6,
        label=f"Best Epoch: {best_epoch}",
    )

    ax.set_title(
        "Transformer Training and Validation Loss",
        fontweight="bold",
        pad=18,
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_xticks(epochs)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path / "03_training_loss.png")
    plt.close(fig)

    # -------------------------------------------------
    # FIGURE 4: Relative Performance Improvement
    # -------------------------------------------------

    baseline = results["N-gram"]
    transformer = results["Transformer"]

    improvement_metrics = [
        "Accuracy@1",
        "Accuracy@5",
        "Accuracy@10",
        "MRR",
    ]

    improvements = [
        (
            transformer[metric] - baseline[metric]
        ) / baseline[metric] * 100
        for metric in improvement_metrics
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(
        improvement_metrics,
        improvements,
    )

    ax.bar_label(
        bars,
        fmt="+%.1f%%",
        padding=5,
        fontsize=10,
    )

    ax.set_title(
        "Transformer Improvement Over N-gram",
        fontweight="bold",
        pad=18,
    )

    ax.set_xlabel("Relative Improvement (%)")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(output_path / "04_relative_improvement.png")
    plt.close(fig)

    print("\n" + "=" * 55)
    print("VISUALIZATION GENERATION COMPLETE")
    print("=" * 55)

    for file in sorted(output_path.glob("*.png")):
        print("Saved:", file)

    print("\nTotal graphs generated: 4")

    return [
        str(file)
        for file in sorted(output_path.glob("*.png"))
    ]
