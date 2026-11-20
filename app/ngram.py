
"""N-gram model for retail basket-completion prediction."""

from collections import Counter, defaultdict
import math


class NGramModel:
    """Predict the next product using product-sequence context."""

    def __init__(self, n=3, alpha=0.1):
        if n < 1:
            raise ValueError("n must be at least 1")

        self.n = n
        self.alpha = alpha
        self.context_counts = defaultdict(Counter)
        self.unigram_counts = Counter()
        self.vocabulary = []

    def fit(self, baskets):
        """Learn product frequencies and context counts from training baskets."""

        self.context_counts.clear()
        self.unigram_counts.clear()

        vocabulary = set()

        for basket in baskets:
            items = list(dict.fromkeys(basket))

            for item in items:
                vocabulary.add(item)
                self.unigram_counts[item] += 1

            for index, item in enumerate(items):
                context = tuple(items[max(0, index - self.n + 1):index])
                self.context_counts[context][item] += 1

        self.vocabulary = sorted(vocabulary)
        return self

    
    def predict_scores(self, visible):
        """Return smoothed log-probability scores for every known product."""

        if not self.vocabulary:
            raise ValueError("Train the model before prediction")

        visible = list(dict.fromkeys(visible))

        context = tuple(visible[-(self.n - 1):]) if self.n > 1 else ()

        # Use unigram frequencies for n=1.
        if self.n == 1:
            counts = self.unigram_counts
            total = sum(counts.values())

        else:
            # Back off from trigram to bigram to unigram
            # when a context has not been observed.
            counts = None

            for length in range(len(context), 0, -1):
                candidate = context[-length:]
                if candidate in self.context_counts:
                    counts = self.context_counts[candidate]
                    break

            if counts is None:
                counts = self.unigram_counts

            total = sum(counts.values())

        scores = {}

        for item in self.vocabulary:
            probability = (
                counts[item] + self.alpha
            ) / (
                total + self.alpha * len(self.vocabulary)
            )

            scores[item] = math.log(probability)

        return scores


    def predict(self, visible, k=5):
        """Return the top-k product recommendations, excluding visible products."""

        scores = self.predict_scores(visible)

        visible_set = set(visible)

        ranked = sorted(
            (
                (item, score)
                for item, score in scores.items()
                if item not in visible_set
            ),
            key=lambda pair: pair[1],
            reverse=True,
        )

        return ranked[:k]
