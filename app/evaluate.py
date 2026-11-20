
"""Fair evaluation of n-gram and Transformer basket predictors."""

import json
import random

import numpy as np
import torch

from thilimarketbasket.dataset import load_dataset

from app.ngram import NGramModel
from app.tokenizer import BasketTokenizer
from app.language_model import BasketTransformer


def evaluate_models(
    dataset_dir="data/retail_dataset",
    model_dir="models/basket_transformer",
    n=3,
    seed=42,
):
    """Compare both models using identical held-out test cases."""

    dataset = load_dataset(dataset_dir)

    # Reconstruct the exact train, validation and test split.
    indices = np.random.RandomState(seed).permutation(
        len(dataset.baskets)
    )

    train_end = int(0.80 * len(indices))
    val_end = int(0.90 * len(indices))

    train_indices = indices[:train_end]
    test_indices = indices[val_end:]

    baskets = [
        sorted(set(basket))
        for basket in dataset.baskets
    ]

    train_baskets = [baskets[i] for i in train_indices]
    test_baskets = [baskets[i] for i in test_indices]

    # Reproduce the randomized training sequences.
    rng = random.Random(seed)
    train_sequences = []

    for basket in train_baskets:
        items = list(basket)
        rng.shuffle(items)
        train_sequences.append(items)

    # Train n-gram on the same training baskets.
    ngram = NGramModel(n=n)
    ngram.fit(train_sequences)

    # Load the Transformer and its tokenizer.
    with open(
        f"{model_dir}/tokenizer.json",
        "r",
        encoding="utf-8",
    ) as file:
        saved_tokens = json.load(file)

    tokenizer = BasketTokenizer(
        [item for item in saved_tokens if not item.startswith("<")]
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        f"{model_dir}/best_model.pt",
        map_location=device,
        weights_only=True,
    )

    transformer = BasketTransformer(
        vocab_size=checkpoint["vocab_size"],
        embedding_dim=checkpoint["embedding_dim"],
        num_heads=checkpoint["num_heads"],
        num_layers=checkpoint["num_layers"],
        max_length=checkpoint["max_length"],
    ).to(device)

    transformer.load_state_dict(
        checkpoint["model_state_dict"]
    )

    transformer.eval()

    # Use the same candidate products for both models.
    candidates = [
        item for item in ngram.vocabulary
        if item in tokenizer.token_to_id
    ]

    
    # Create identical test cases for both models.
    test_rng = random.Random(seed + 2)

    ranks = {
        "N-gram": [],
        "Transformer": [],
    }

    skipped = 0

    for basket in test_baskets:
        items = list(basket)
        test_rng.shuffle(items)

        if len(items) < 2:
            skipped += 1
            continue

        visible = items[:-1]
        target = items[-1]

        case_candidates = [
            item for item in candidates
            if item not in visible
        ]

        if target not in case_candidates:
            skipped += 1
            continue

        # N-gram ranking.
        ngram_scores = ngram.predict_scores(visible)

        ngram_ranking = sorted(
            case_candidates,
            key=lambda item: ngram_scores[item],
            reverse=True,
        )

        ngram_rank = ngram_ranking.index(target) + 1

        ranks["N-gram"].append(ngram_rank)

        # Transformer ranking.
        input_ids = [
            tokenizer.bos_id
        ] + [
            tokenizer.token_to_id[item]
            for item in visible
        ]

        input_tensor = torch.tensor(
            [input_ids],
            dtype=torch.long,
            device=device,
        )

        with torch.no_grad():
            logits = transformer(input_tensor)[0, -1]

            case_candidate_ids = torch.tensor(
                [tokenizer.token_to_id[item] for item in case_candidates],
                device=device,
            )

            candidate_scores = logits[case_candidate_ids]

            transformer_order = torch.argsort(
                candidate_scores,
                descending=True,
            )

            transformer_ranking = case_candidate_ids[
                transformer_order
            ].tolist()

        transformer_rank = (
            transformer_ranking.index(
                tokenizer.token_to_id[target]
            ) + 1
        )

        ranks["Transformer"].append(transformer_rank)

    # Calculate ranking metrics.
    results = {}

    for model_name, model_ranks in ranks.items():
        rank_array = np.array(model_ranks)

        results[model_name] = {
            "Accuracy@1": np.mean(rank_array <= 1),
            "Accuracy@5": np.mean(rank_array <= 5),
            "Accuracy@10": np.mean(rank_array <= 10),
            "MRR": np.mean(1 / rank_array),
            "Mean Rank": np.mean(rank_array),
            "Evaluated Cases": len(rank_array),
        }

    print("\n" + "=" * 60)
    print("RETAIL BASKET MODEL EVALUATION")
    print("=" * 60)

    print("Training baskets:", len(train_baskets))
    print("Test baskets:", len(test_baskets))
    print("Evaluated cases:", len(ranks["N-gram"]))
    print("Skipped cases:", skipped)
    print("Candidate products:", len(candidates))
    print("Evaluation device:", device)

    for model_name, metrics in results.items():
        print("\n" + model_name)

        for metric, value in metrics.items():
            if isinstance(value, (int, np.integer)):
                print(f"{metric}: {value}")
            else:
                print(f"{metric}: {value:.4f}")

        # Save evaluation results for visualization and reporting.
    output_path = "results"

    import os
    os.makedirs(output_path, exist_ok=True)

    with open(
        os.path.join(output_path, "evaluation_results.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(results, file, indent=4)

    print("\nEvaluation results saved to:")
    print("results/evaluation_results.json")

    return results