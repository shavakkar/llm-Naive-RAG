from typing import List
import ollama
from ai.embeddings.base import BaseEmbedder

class OllamaEmbedder(BaseEmbedder):
    """
    Ollama REST Client Execution.
    """
    def __init__(self, model_name: str, base_url: str):
        print(f"[Embedder] Connecting to Ollama Embedding Service ({model_name})...")
        self.model_name = model_name
        self.client = ollama.Client(host=base_url)

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        response = self.client.embed(model=self.model_name, input=documents)
        return response["embeddings"]

    def embed_query(self, query: str) -> List[List[float]]:
        response = self.client.embed(model=self.model_name, input=[query])
        return response["embeddings"]