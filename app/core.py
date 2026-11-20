
"""Core functionality for the Retail Basket Prediction Analyzer."""

import random
from pathlib import Path

import numpy as np

from thilimarketbasket.dataset import (
    generate_dataset,
    save_dataset,
    load_dataset,
)

from app.ngram import NGramModel


def milestone_1(
    n_baskets=20_000,
    seed=42,
    min_support=0.0005,
    max_size=5,
    output_dir="data/retail_dataset",
):
    """Generate and save synthetic shopping baskets."""

    dataset = generate_dataset(
        n_baskets=n_baskets,
        seed=seed,
        min_support=min_support,
        max_size=max_size,
    )

    save_dataset(dataset, Path(output_dir))

    return {
        "n_baskets": len(dataset.baskets),
        "n_products": len(dataset.catalog),
        "n_planted_itemsets": len(dataset.planted),
        "n_frequent_itemsets": len(dataset.truth),
        "mean_basket_size": sum(
            len(set(basket)) for basket in dataset.baskets
        ) / len(dataset.baskets),
        "seed": seed,
        "min_support": min_support,
        "output_dir": output_dir,
    }


def milestone_2(
    dataset_dir="data/retail_dataset",
    n=3,
    alpha=0.1,
    seed=42,
):
    """Train the n-gram model using the shared training split."""

    dataset = load_dataset(dataset_dir)

    baskets = [
        sorted(set(basket))
        for basket in dataset.baskets
    ]

    indices = np.random.RandomState(seed).permutation(
        len(baskets)
    )

    train_end = int(0.80 * len(indices))
    train_indices = indices[:train_end]

    train_baskets = [
        baskets[i] for i in train_indices
    ]

    rng = random.Random(seed)
    train_sequences = []

    for basket in train_baskets:
        items = list(basket)
        rng.shuffle(items)
        train_sequences.append(items)

    model = NGramModel(n=n, alpha=alpha)
    model.fit(train_sequences)

    return {
        "model": model,
        "n": n,
        "vocabulary_size": len(model.vocabulary),
        "training_baskets": len(train_baskets),
        "total_baskets": len(baskets),
    }


def milestone_3(
    dataset_dir="data/retail_dataset",
    output_dir="models/basket_transformer",
    epochs=5,
    batch_size=16,
):
    """Train the PyTorch Transformer language model."""

    from app.train import train_language_model

    return train_language_model(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
    )


def milestone_4(
    dataset_dir="data/retail_dataset",
    model_dir="models/basket_transformer",
    n=3,
):
    """Evaluate the n-gram and Transformer models."""

    from app.evaluate import evaluate_models

    return evaluate_models(
        dataset_dir=dataset_dir,
        model_dir=model_dir,
        n=n,
    )


def milestone_5(
    results_file="results/evaluation_results.json",
    history_file="models/basket_transformer/training_history.json",
    output_dir="results/figures",
):
    """Generate professional model comparison visualizations."""

    from app.visualize import create_visualizations

    return create_visualizations(
        results_file=results_file,
        history_file=history_file,
        output_dir=output_dir,
    )
