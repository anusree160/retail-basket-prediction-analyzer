
"""Core functionality for the Retail Basket Prediction Analyzer."""

from pathlib import Path

from thilimarketbasket.dataset import generate_dataset, save_dataset, load_dataset
from app.ngram import NGramModel


def milestone_1(
    n_baskets: int = 20_000,
    seed: int = 42,
    min_support: float = 0.0005,
    max_size: int = 5,
    output_dir: str = "data/retail_dataset",
):
    """Generate and save synthetic shopping baskets with known purchasing patterns."""

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
):
    """Train an n-gram model on the generated retail basket dataset."""

    dataset = load_dataset(dataset_dir)

    model = NGramModel(n=n, alpha=alpha)
    model.fit(dataset.baskets)

    return {
        "model": model,
        "n": n,
        "vocabulary_size": len(model.vocabulary),
        "training_baskets": len(dataset.baskets),
    }


def milestone_3():
    raise NotImplementedError("TODO(milestone 3)")


def milestone_4():
    raise NotImplementedError("TODO(milestone 4)")


def milestone_5():
    raise NotImplementedError("TODO(milestone 5)")
