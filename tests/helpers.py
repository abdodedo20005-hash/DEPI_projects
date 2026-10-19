"""Small helpers shared by the tests."""

import hashlib
import re

import numpy as np


def hash_embedder(texts):
    """Stand-in for the embedding model: hashed bag of words, normalised. Matches WORDS, not meaning."""
    vectors = np.zeros((len(texts), 512), dtype="float32")
    for row, text in enumerate(texts):
        for word in re.findall(r"\w+", text.lower()):
            vectors[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % 512] += 1
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms
