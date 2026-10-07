from typing import Dict, Any, List
from ai.config import Config, ExecutionProvider
from ai.embeddings.base import BaseEmbedder
from ai.embeddings.local_adapter import LocalQwenEmbedder
from ai.embeddings.ollama_adapter import OllamaEmbedder
from ai.agents.base import BaseGenerator
from ai.agents.local_adapter import LocalDeepSeekGenerator
from ai.agents.ollama_adapter import OllamaGenerator
from ai.rag.chroma_client import ChromaVectorStore

class RAGPipelineFactory:
    """
    Factory constructing RAG components based on system configuration.
    """
    @staticmethod
    def create_embedder() -> BaseEmbedder:
        if Config.PROVIDER == ExecutionProvider.LOCAL:
            return LocalQwenEmbedder(model_path=Config.LOCAL_EMBED_PATH)
        elif Config.PROVIDER == ExecutionProvider.OLLAMA:
            return OllamaEmbedder(
                model_name=Config.OLLAMA_EMBED_MODEL,
                base_url=Config.OLLAMA_BASE_URL
            )
        else:
            raise ValueError(f"Unsupported execution provider: {Config.PROVIDER}")

    @staticmethod
    def create_generator() -> BaseGenerator:
        if Config.PROVIDER == ExecutionProvider.LOCAL:
            return LocalDeepSeekGenerator(model_path=Config.LOCAL_LLM_PATH)
        elif Config.PROVIDER == ExecutionProvider.OLLAMA:
            return OllamaGenerator(
                model_name=Config.OLLAMA_LLM_MODEL,
                base_url=Config.OLLAMA_BASE_URL
            )
        else:
            raise ValueError(f"Unsupported execution provider: {Config.PROVIDER}")

class RAGPipeline:
    """
    Orchestrates ingestion, retrieval, and generation.
    """
    def __init__(self, embedder: BaseEmbedder, generator: BaseGenerator, vector_store: ChromaVectorStore):
        self.embedder = embedder
        self.generator = generator
        self.vector_store = vector_store

    def ingest_documents(self, documents: List[str]) -> None:
        print(f"\n[Pipeline] Embedding {len(documents)} document(s)...")
        embeddings = self.embedder.embed_documents(documents)
        self.vector_store.add_documents(documents=documents, embeddings=embeddings)
        print("[Pipeline] Ingestion completed successfully.")

    def query(self, user_query: str) -> Dict[str, Any]:
        print(f"\n[Pipeline] Searching context for query: '{user_query}'")
        query_vector = self.embedder.embed_query(user_query)
        retrieved_docs = self.vector_store.query(
            query_embedding=query_vector,
            top_k=Config.TOP_K_RESULTS
        )
        
        print(f"[Pipeline] Retrieved {len(retrieved_docs)} relevant context passage(s).")
        result = self.generator.generate_answer(query=user_query, context_documents=retrieved_docs)
        result["retrieved_docs"] = retrieved_docs
        return result