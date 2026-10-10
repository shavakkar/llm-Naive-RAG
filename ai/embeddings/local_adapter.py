from typing import List
import torch
from sentence_transformers import SentenceTransformer
from ai.embeddings.base import BaseEmbedder

class LocalQwenEmbedder(BaseEmbedder):
    """
    Local PyTorch / SentenceTransformers execution.
    """
    def __init__(self, model_path: str):
        print(f"[Embedder] Loading Local Hugging Face model from: {model_path}")
        self.model = SentenceTransformer(
            model_path,
            trust_remote_code=True,
            model_kwargs={"dtype": torch.float32}
        )

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(documents, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[List[float]]:
        # Encodes with Qwen3's specific search prompt alignment
        embeddings = self.model.encode([query], prompt_name="query", normalize_embeddings=True)
        return embeddings.tolist()