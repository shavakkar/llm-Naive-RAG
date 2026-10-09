import uuid
from typing import List, Dict, Any
import chromadb

class ChromaVectorStore:
    """
    Encapsulates database mechanics for ChromaDB storage, including
    metadata tracking and distance scoring.
    """
    def __init__(self, persist_dir: str, collection_name: str):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        """
        Expects chunked document dictionaries:
        [{"content": "...", "metadata": {"source": "...", "chunk_id": 0}}, ...]
        """
        if not documents:
            return

        ids = [str(uuid.uuid4()) for _ in documents]
        texts = [doc["content"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]

        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings
        )
        print(f"[ChromaVectorStore] Stored {len(documents)} chunk(s) with metadata in ChromaDB.")

    def query(self, query_embedding: List[List[float]], top_k: int) -> List[Dict[str, Any]]:
        """
        Returns structured search results including text, metadata, and distance scores.
        """
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        processed_results = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if "metadatas" in results and results["metadatas"] else [{}] * len(docs)
            dists = results["distances"][0] if "distances" in results and results["distances"] else [0.0] * len(docs)

            for doc_text, meta, dist in zip(docs, metas, dists):
                processed_results.append({
                    "content": doc_text,
                    "metadata": meta,
                    "distance": round(dist, 4)  # Lower score = higher vector similarity
                })

        return processed_results