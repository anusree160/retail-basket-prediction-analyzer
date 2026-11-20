
"""Automated tests for the five project milestones."""

import pytest
import json

from app.core import milestone_1


# Milestone 1: Generate a synthetic dataset of shopping baskets.

def test_milestone_1(tmp_path):
    output_dir = tmp_path / "retail_dataset"

    result = milestone_1(
        n_baskets=10_000,
        seed=42,
        output_dir=str(output_dir),
    )

    # Verify generated dataset statistics.
    assert result["n_baskets"] == 10_000
    assert result["n_products"] == 1000
    assert result["n_planted_itemsets"] == 32
    assert result["n_frequent_itemsets"] > 0

    # Verify that all expected dataset files exist.
    expected_files = [
        "baskets.jsonl",
        "catalog.json",
        "planted.json",
        "truth.json",
        "meta.json",
    ]

    for filename in expected_files:
        assert (output_dir / filename).exists()

    # Verify saved metadata.
    with open(output_dir / "meta.json", encoding="utf-8") as file:
        metadata = json.load(file)

    assert metadata["n_baskets"] == 10_000
    assert metadata["n_planted"] == 32
    assert metadata["seed"] == 42

    # Verify planted patterns.
    with open(output_dir / "planted.json", encoding="utf-8") as file:
        patterns = json.load(file)

    assert len(patterns) == 32


# Milestone 2: Implement the n-gram next-item predictor.

# Milestone 2: Implement the n-gram next-item predictor.

from app.core import milestone_2


def test_milestone_2():
    result = milestone_2(n=3)

    assert result["n"] == 3
    assert result["training_baskets"] == 20_000
    assert result["vocabulary_size"] == 1000

    model = result["model"]

    predictions = model.predict(["Tea", "Sugar"], k=5)

    assert len(predictions) == 5
    assert all(item not in ["Tea", "Sugar"] for item, score in predictions)
    assert all(isinstance(score, float) for item, score in predictions)


# Milestone 3: Pretrain a small language model.

@pytest.mark.skip(reason="not implemented yet")
def test_milestone_3():
    raise NotImplementedError


# Milestone 4: Evaluate both models.

@pytest.mark.skip(reason="not implemented yet")
def test_milestone_4():
    raise NotImplementedError


# Milestone 5: Compare model performance.

@pytest.mark.skip(reason="not implemented yet")
def test_milestone_5():
    raise NotImplementedError
