from abc import ABC, abstractmethod

import numpy as np


class VectorStore(ABC):
    """Minimal interface so backends can be swapped and benchmarked."""

    @abstractmethod
    def add(self, entry_id: int, namespace: str, embedding: np.ndarray) -> None: ...

    @abstractmethod
    def search(self, namespace: str, embedding: np.ndarray, top_k: int = 3) -> list[tuple[int, float]]:
        """Return [(entry_id, cosine_similarity)] best first."""

    @abstractmethod
    def delete(self, entry_ids: list[int]) -> None: ...

    @abstractmethod
    def clear(self, namespace: str | None = None) -> None: ...
