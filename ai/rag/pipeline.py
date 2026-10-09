import sys
from typing import Dict, Any, List
from ai.config import Config, ExecutionProvider, get_categorized_ollama_models
from ai.embeddings.base import BaseEmbedder
from ai.embeddings.local_adapter import LocalQwenEmbedder
from ai.embeddings.ollama_adapter import OllamaEmbedder
from ai.agents.base import BaseGenerator
from ai.agents.local_adapter import LocalDeepSeekGenerator
from ai.agents.ollama_adapter import OllamaGenerator
from ai.rag.chroma_client import ChromaVectorStore

class RAGPipelineFactory:
    @staticmethod
    def prompt_ollama_model_selection():
        """
        Displays interactive CLI menu for user to pick Ollama models.
        """
        models = get_categorized_ollama_models(Config.OLLAMA_BASE_URL)
        
        if not models["embedders"] and not models["generators"]:
            print("\n❌ No models found in Ollama! Please ensure Ollama is running (`ollama serve`).\n")
            sys.exit(1)

        print("\n" + "=" * 50)
        print("🤖 OLLAMA MODEL SELECTION MENU")
        print("=" * 50)

        # 1. Select Embedder
        selected_embedder = None
        if models["embedders"]:
            print("\nSelect Embedding Model:")
            for idx, model in enumerate(models["embedders"], 1):
                print(f"  [{idx}] {model}")
            choice = int(input("Enter number for Embedding Model: ")) - 1
            selected_embedder = models["embedders"][choice]
        else:
            print("⚠️ No dedicated embedding models found. Falling back to default embedder selection.")

        # 2. Select Generator LLM
        if not models["generators"]:
            print("\n❌ No text generation LLM models found in Ollama!")
            print("👉 Run: `ollama pull qwen2.5:7b` to install an LLM.\n")
            sys.exit(1)

        print("\nSelect LLM Generator Model:")
        for idx, model in enumerate(models["generators"], 1):
            print(f"  [{idx}] {model}")
        choice = int(input("Enter number for LLM Generator: ")) - 1
        selected_generator = models["generators"][choice]

        print("=" * 50 + "\n")
        
        Config.OLLAMA_EMBED_MODEL = selected_embedder
        Config.OLLAMA_LLM_MODEL = selected_generator

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

    def ingest_documents(self, documents: List[Dict[str, Any]]) -> None:
        print(f"\n[Pipeline] Ingesting {len(documents)} chunked document(s)...")
        texts = [doc["content"] for doc in documents]
        embeddings = self.embedder.embed_documents(texts)
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
        context_texts = [doc["content"] for doc in retrieved_docs]
        
        result = self.generator.generate_answer(query=user_query, context_documents=context_texts)
        result["retrieved_docs"] = retrieved_docs
        return result