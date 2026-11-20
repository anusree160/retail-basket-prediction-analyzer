
"""Core functionality for the Retail Basket Prediction Analyzer."""

from pathlib import Path

from thilimarketbasket.dataset import generate_dataset, save_dataset


# --- Milestone 1: Generate a synthetic retail basket dataset.

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


# --- Milestone 2: N-gram next-item prediction.

def milestone_2():
    raise NotImplementedError("TODO(milestone 2)")


# --- Milestone 3: Train a small neural language model.

def milestone_3():
    raise NotImplementedError("TODO(milestone 3)")


# --- Milestone 4: Evaluate both models.

def milestone_4():
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Compare model performance.

def milestone_5():
    raise NotImplementedError("TODO(milestone 5)")
