import uuid
from pathlib import Path
from typing import List, Dict, Any
import chromadb

class ChromaVectorStore:
    """
    Encapsulates database mechanics for ChromaDB storage, including
    automatic directory creation, metadata tracking, and distance scoring.
    """
    def __init__(self, persist_dir: str, collection_name: str):
        self.persist_path = Path(persist_dir).resolve()
        
        # 1. Force directory creation inside project folder
        self.persist_path.mkdir(parents=True, exist_ok=True)
        print(f"[ChromaVectorStore] Database path: '{self.persist_path}'")

        # 2. Connect persistent client using absolute path string
        self.client = chromadb.PersistentClient(path=str(self.persist_path))
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        """
        Stores document chunks and embeddings. Force-persists to disk.
        """
        if not documents or not embeddings:
            print("⚠️ [ChromaVectorStore] No documents or embeddings to store.")
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
        print(f"[ChromaVectorStore] Successfully persisted {len(documents)} chunk(s) to '{self.persist_path}'.")

    def query(self, query_embedding: List[List[float]], top_k: int) -> List[Dict[str, Any]]:
        """
        Returns search results. Handles empty collections safely.
        """
        # Guard against querying an empty collection
        if self.collection.count() == 0:
            print("⚠️ [ChromaVectorStore] Vector store collection is currently empty!")
            return []

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, self.collection.count()),  # Avoid requesting more than exists
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
                    "distance": round(dist, 4)
                })

        return processed_results