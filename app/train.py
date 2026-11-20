
"""Train the retail basket Transformer using PyTorch."""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from thilimarketbasket.dataset import load_dataset

from app.tokenizer import BasketTokenizer
from app.language_model import BasketTransformer


def train_language_model(
    dataset_dir="data/retail_dataset",
    output_dir="models/basket_transformer",
    epochs=5,
    batch_size=16,
    learning_rate=0.0003,
    seed=42,
):
    """Train a small causal Transformer on retail basket sequences."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Loading retail dataset...")
    dataset = load_dataset(dataset_dir)

    product_names = [item.name for item in dataset.catalog]
    tokenizer = BasketTokenizer(product_names)

    # Remove duplicate products from each basket.
    baskets = [
        sorted(set(basket))
        for basket in dataset.baskets
    ]

    # Shuffle baskets and split before training.
    indices = np.random.permutation(len(baskets))

    train_end = int(0.80 * len(indices))
    val_end = int(0.90 * len(indices))

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    train_baskets = [baskets[i] for i in train_indices]
    val_baskets = [baskets[i] for i in val_indices]
    test_baskets = [baskets[i] for i in test_indices]

    print("Training baskets:", len(train_baskets))
    print("Validation baskets:", len(val_baskets))
    print("Test baskets:", len(test_baskets))
    print("Vocabulary size:", tokenizer.vocabulary_size)
    print("Training device:", device)

    max_length = 64

    def encode_baskets(basket_list, shuffle_seed):
        rng = random.Random(shuffle_seed)
        sequences = []

        for basket in basket_list:
            items = list(basket)
            rng.shuffle(items)

            sequences.append(
                tokenizer.encode(items)[:max_length]
            )

        return sequences

    train_sequences = encode_baskets(train_baskets, seed)
    val_sequences = encode_baskets(val_baskets, seed + 1)

    def make_loader(sequences, shuffle):
        input_data = []
        target_data = []
        attention_masks = []

        for sequence in sequences:
            inputs = sequence[:-1]
            targets = sequence[1:]

            padding = max_length - len(inputs)

            input_data.append(
                inputs + [tokenizer.pad_id] * padding
            )

            target_data.append(
                targets + [-100] * padding
            )

            attention_masks.append(
                [1] * len(inputs) + [0] * padding
            )

        inputs = torch.tensor(input_data, dtype=torch.long)
        targets = torch.tensor(target_data, dtype=torch.long)
        masks = torch.tensor(attention_masks, dtype=torch.bool)

        dataset_tensor = TensorDataset(inputs, targets, masks)

        return DataLoader(
            dataset_tensor,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=(device.type == "cuda"),
        )

    train_loader = make_loader(train_sequences, True)
    val_loader = make_loader(val_sequences, False)

    model = BasketTransformer(
        vocab_size=tokenizer.vocabulary_size,
        embedding_dim=128,
        num_heads=4,
        num_layers=2,
        max_length=max_length,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )

    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    use_amp = device.type == "cuda"

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    def evaluate():
        model.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for inputs, targets, masks in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                masks = masks.to(device)

                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    logits = model(inputs)

                    loss = criterion(
                        logits.reshape(-1, tokenizer.vocabulary_size),
                        targets.reshape(-1),
                    )

                valid_tokens = masks.sum().item()

                total_loss += loss.item() * valid_tokens
                total_tokens += valid_tokens

        return total_loss / max(total_tokens, 1)

    history = []
    best_val_loss = float("inf")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\nStarting Transformer training...\n")

    started = time.time()

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.0
        total_tokens = 0

        for inputs, targets, masks in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(inputs)

                loss = criterion(
                    logits.reshape(-1, tokenizer.vocabulary_size),
                    targets.reshape(-1),
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            scaler.step(optimizer)
            scaler.update()

            valid_tokens = masks.sum().item()

            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

        train_loss = total_loss / max(total_tokens, 1)
        val_loss = evaluate()

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }

        history.append(record)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Validation Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab_size": tokenizer.vocabulary_size,
                    "max_length": max_length,
                    "embedding_dim": 128,
                    "num_heads": 4,
                    "num_layers": 2,
                },
                output_path / "best_model.pt",
            )

            print("  Best model checkpoint saved.")

    with open(
        output_path / "tokenizer.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(tokenizer.id_to_token, file, indent=2)

    with open(
        output_path / "training_history.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(history, file, indent=2)

    with open(
        output_path / "split_indices.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "train": train_indices.tolist(),
                "validation": val_indices.tolist(),
                "test": test_indices.tolist(),
                "seed": seed,
            },
            file,
        )

    print("\nTraining completed.")
    print("Best validation loss:", round(best_val_loss, 4))
    print("Training time:", round(time.time() - started, 2), "seconds")
    print("Saved model:", output_path / "best_model.pt")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "history": history,
        "best_val_loss": best_val_loss,
        "test_baskets": test_baskets,
    }
