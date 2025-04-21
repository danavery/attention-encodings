import numpy as np


class NumpyIndexFlatIP:
    """
    Drop-in numpy-only replacement for FAISS as it's used in attention-encodings.
    Hopefully useful for stopping segfaults on macs due to interaction between
    FAISS and the transformers library there.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.vectors = None

    def add(self, vectors: np.ndarray):
        vectors = np.asarray(vectors)
        assert vectors.shape[1] == self.dim, f"Expected vectors with dim={self.dim}"
        self.vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def search(self, queries: np.ndarray, k: int):
        if self.vectors is None:
            raise RuntimeError("Index is empty. Call `.add()` before `.search()`.")

        queries = np.asarray(queries)
        if queries.ndim == 1:
            queries = queries[np.newaxis, :]

        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)

        sims = np.dot(queries, self.vectors.T)  # shape: (n_queries, vocab_size)
        indices = np.argsort(-sims, axis=1)[:, :k]  # sort descending
        sorted_sims = np.take_along_axis(sims, indices, axis=1)

        return sorted_sims, indices
