
"""Tokenizer for retail basket product sequences."""


class BasketTokenizer:
    """Convert product names into integer token IDs."""

    PAD_TOKEN = "<PAD>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"

    def __init__(self, product_names):
        self.special_tokens = [
            self.PAD_TOKEN,
            self.BOS_TOKEN,
            self.EOS_TOKEN,
            self.UNK_TOKEN,
        ]

        products = sorted(set(product_names))

        self.id_to_token = self.special_tokens + products

        self.token_to_id = {
            token: index
            for index, token in enumerate(self.id_to_token)
        }

        self.pad_id = self.token_to_id[self.PAD_TOKEN]
        self.bos_id = self.token_to_id[self.BOS_TOKEN]
        self.eos_id = self.token_to_id[self.EOS_TOKEN]
        self.unk_id = self.token_to_id[self.UNK_TOKEN]

    @property
    def vocabulary_size(self):
        return len(self.id_to_token)

    def encode(self, basket):
        """Convert a basket into token IDs."""

        unique_items = list(dict.fromkeys(basket))

        token_ids = [
            self.token_to_id.get(item, self.unk_id)
            for item in unique_items
        ]

        return [self.bos_id] + token_ids + [self.eos_id]

    def decode(self, token_ids):
        """Convert token IDs back into product names."""

        special = set(self.special_tokens)

        return [
            self.id_to_token[token_id]
            for token_id in token_ids
            if 0 <= token_id < len(self.id_to_token)
            and self.id_to_token[token_id] not in special
        ]
