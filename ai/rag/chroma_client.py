import uuid
from typing import List, Dict, Any
import chromadb

class ChromaVectorStore:
    """
    Encapsulates database mechanics for ChromaDB storage.
    """
    def __init__(self, persist_dir: str, collection_name: str):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_documents(self, documents: List[str], embeddings: List[List[float]]) -> None:
        ids = [str(uuid.uuid4()) for _ in documents]
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings
        )

    def query(self, query_embedding: List[List[float]], top_k: int) -> List[str]:
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        if results and "documents" in results and results["documents"]:
            return results["documents"][0]
        return []