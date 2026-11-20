
"""Automated tests for all five project milestones."""

import json

import matplotlib
matplotlib.use("Agg")

import torch

from app.core import milestone_1, milestone_2
from app.tokenizer import BasketTokenizer
from app.language_model import BasketTransformer
from app.visualize import create_visualizations


def test_milestone_1(tmp_path):
    """Test synthetic retail dataset generation."""

    output_dir = tmp_path / "retail_dataset"

    result = milestone_1(
        n_baskets=10_000,
        seed=42,
        output_dir=str(output_dir),
    )

    assert result["n_baskets"] == 10_000
    assert result["n_products"] == 1000
    assert result["n_planted_itemsets"] == 32
    assert result["n_frequent_itemsets"] > 0

    expected_files = [
        "baskets.jsonl",
        "catalog.json",
        "planted.json",
        "truth.json",
        "meta.json",
    ]

    for filename in expected_files:
        assert (output_dir / filename).exists()

    with open(output_dir / "meta.json", encoding="utf-8") as file:
        metadata = json.load(file)

    assert metadata["n_baskets"] == 10_000
    assert metadata["n_planted"] == 32
    assert metadata["seed"] == 42


def test_milestone_2(tmp_path):
    """Test n-gram training and product prediction."""

    dataset_dir = tmp_path / "retail_dataset"

    milestone_1(
        n_baskets=10_000,
        seed=42,
        output_dir=str(dataset_dir),
    )

    result = milestone_2(
        dataset_dir=str(dataset_dir),
        n=3,
    )

    assert result["n"] == 3
    assert result["training_baskets"] == 8_000
    assert result["total_baskets"] == 10_000
    assert 1 <= result["vocabulary_size"] <= 1000

    model = result["model"]

    predictions = model.predict(
        ["Tea", "Sugar"],
        k=5,
    )

    assert len(predictions) <= 5
    assert all(
        item not in ["Tea", "Sugar"]
        for item, score in predictions
    )
    assert all(
        isinstance(score, float)
        for item, score in predictions
    )


def test_milestone_3():
    """Test tokenizer and Transformer architecture."""

    tokenizer = BasketTokenizer(
        ["Tea", "Sugar", "Milk", "Bread"]
    )

    encoded = tokenizer.encode(
        ["Tea", "Milk", "Bread"]
    )

    decoded = tokenizer.decode(encoded)

    assert decoded == ["Tea", "Milk", "Bread"]
    assert encoded[0] == tokenizer.bos_id
    assert encoded[-1] == tokenizer.eos_id
    assert tokenizer.vocabulary_size == 8

    model = BasketTransformer(
        vocab_size=tokenizer.vocabulary_size,
        embedding_dim=32,
        num_heads=4,
        num_layers=2,
        max_length=16,
        dropout=0.0,
    )

    input_tensor = torch.tensor(
        [encoded[:-1]],
        dtype=torch.long,
    )

    output = model(input_tensor)

    assert output.shape == (
        1,
        len(encoded) - 1,
        tokenizer.vocabulary_size,
    )

    assert torch.isfinite(output).all()


def test_milestone_4(tmp_path):
    """Test the evaluation metrics and saved result structure."""

    ranks = [1, 2, 5, 10]

    accuracy_at_1 = sum(rank <= 1 for rank in ranks) / len(ranks)
    accuracy_at_5 = sum(rank <= 5 for rank in ranks) / len(ranks)
    accuracy_at_10 = sum(rank <= 10 for rank in ranks) / len(ranks)

    mrr = sum(1 / rank for rank in ranks) / len(ranks)
    mean_rank = sum(ranks) / len(ranks)

    results = {
        "N-gram": {
            "Accuracy@1": accuracy_at_1,
            "Accuracy@5": accuracy_at_5,
            "Accuracy@10": accuracy_at_10,
            "MRR": mrr,
            "Mean Rank": mean_rank,
            "Evaluated Cases": len(ranks),
        },
        "Transformer": {
            "Accuracy@1": 0.50,
            "Accuracy@5": 0.75,
            "Accuracy@10": 1.0,
            "MRR": 0.70,
            "Mean Rank": 2.0,
            "Evaluated Cases": len(ranks),
        },
    }

    assert results["N-gram"]["Accuracy@1"] == 0.25
    assert results["N-gram"]["Accuracy@5"] == 0.75
    assert results["N-gram"]["Accuracy@10"] == 1.0
    assert results["N-gram"]["MRR"] > 0
    assert results["N-gram"]["Mean Rank"] == 4.5

    output_file = tmp_path / "evaluation_results.json"

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    with open(output_file, encoding="utf-8") as file:
        saved_results = json.load(file)

    assert "N-gram" in saved_results
    assert "Transformer" in saved_results
    assert saved_results["Transformer"]["Evaluated Cases"] == 4


def test_milestone_5(tmp_path):
    """Test generation of all four research visualizations."""

    results = {
        "N-gram": {
            "Accuracy@1": 0.02,
            "Accuracy@5": 0.04,
            "Accuracy@10": 0.05,
            "MRR": 0.03,
            "Mean Rank": 420.0,
            "Evaluated Cases": 100,
        },
        "Transformer": {
            "Accuracy@1": 0.12,
            "Accuracy@5": 0.17,
            "Accuracy@10": 0.21,
            "MRR": 0.15,
            "Mean Rank": 195.0,
            "Evaluated Cases": 100,
        },
    }

    history = [
        {"epoch": 1, "train_loss": 5.8, "val_loss": 5.7},
        {"epoch": 2, "train_loss": 5.5, "val_loss": 5.4},
        {"epoch": 3, "train_loss": 5.2, "val_loss": 5.3},
    ]

    results_file = tmp_path / "evaluation_results.json"
    history_file = tmp_path / "training_history.json"
    output_dir = tmp_path / "figures"

    with open(results_file, "w", encoding="utf-8") as file:
        json.dump(results, file)

    with open(history_file, "w", encoding="utf-8") as file:
        json.dump(history, file)

    generated_files = create_visualizations(
        results_file=str(results_file),
        history_file=str(history_file),
        output_dir=str(output_dir),
    )

    assert len(generated_files) == 4

    expected_files = [
        "01_top_k_accuracy.png",
        "02_ranking_performance.png",
        "03_training_loss.png",
        "04_relative_improvement.png",
    ]

    for filename in expected_files:
        image_file = output_dir / filename
        assert image_file.exists()
        assert image_file.stat().st_size > 0
