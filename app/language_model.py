
"""Small GPT-style Transformer for retail basket sequences."""

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    """One Transformer block with attention and feed-forward layers."""

    def __init__(self, embedding_dim=128, num_heads=4, dropout=0.1):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask):
        attention_output, _ = self.attention(
            x, x, x,
            attn_mask=causal_mask,
            need_weights=False,
        )

        x = self.norm1(x + self.dropout(attention_output))
        x = self.norm2(x + self.feed_forward(x))

        return x


class BasketTransformer(nn.Module):
    """Predict the next token in a retail basket sequence."""

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        num_heads=4,
        num_layers=2,
        max_length=64,
        dropout=0.1,
    ):
        super().__init__()

        self.max_length = max_length

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_length, embedding_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(embedding_dim)
        self.output_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, token_ids):
        batch_size, sequence_length = token_ids.shape

        if sequence_length > self.max_length:
            raise ValueError("Sequence exceeds maximum model length")

        positions = torch.arange(
            sequence_length,
            device=token_ids.device,
        )

        x = (
            self.token_embedding(token_ids)
            + self.position_embedding(positions)
        )

        # Prevent each position from attending to future tokens.
        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=token_ids.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        for block in self.blocks:
            x = block(x, causal_mask)

        x = self.final_norm(x)

        return self.output_head(x)
