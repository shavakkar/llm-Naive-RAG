from abc import ABC, abstractmethod
from typing import List

class BaseEmbedder(ABC):
    """
    Abstract Interface for all Embedding Providers.
    """
    @abstractmethod
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Encode a list of text documents into floating-point vectors."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> List[List[float]]:
        """Encode a single search query into floating-point vectors."""
        pass