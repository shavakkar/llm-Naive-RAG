from typing import List
import ollama
from ai.embeddings.base import BaseEmbedder

class OllamaEmbedder(BaseEmbedder):
    """
    Ollama REST Client Execution with graceful model detection.
    """
    def __init__(self, model_name: str | None, base_url: str):
        self.client = ollama.Client(host=base_url)
        self.model_name = self._resolve_model(model_name)
        print(f"[Embedder] Connected to Ollama Embedding Service using model: '{self.model_name}'")

    def _resolve_model(self, specified_model: str | None) -> str:
        try:
            installed_response = self.client.list()
            installed_models = [
                m.get("model") if isinstance(m, dict) else getattr(m, "model", getattr(m, "name", str(m)))
                for m in installed_response.get("models", [])
            ]
        except Exception as e:
            raise ConnectionError(
                f"\n❌ Could not connect to Ollama service at '{self.client._host}'.\n"
                f"   Please ensure Ollama is running (`ollama serve`). Details: {e}"
            )

        if not installed_models:
            raise RuntimeError(
                "\n❌ No models found in your local Ollama instance.\n"
                "   Please install an embedding model by running:\n"
                "   👉 ollama pull nomic-embed-text\n"
                "   (or run: `ollama pull qwen3-embedding`)"
            )

        # 1. If user specified a model, check if it's installed
        if specified_model:
            if any(specified_model in m for m in installed_models):
                return specified_model
            print(f"⚠️ Specified embed model '{specified_model}' not found in installed models.")

        # 2. Look for common embedding models among installed models
        embedding_keywords = ["embed", "bge", "nomic", "minilm"]
        for model in installed_models:
            if any(kw in model.lower() for kw in embedding_keywords):
                return model

        # 3. Fallback to the first available model if no specific embed keywords match
        fallback = installed_models[0]
        print(f"ℹ️ No dedicated embedding model detected. Falling back to installed model: '{fallback}'")
        return fallback

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        response = self.client.embed(model=self.model_name, input=documents)
        return response["embeddings"]

    def embed_query(self, query: str) -> List[List[float]]:
        response = self.client.embed(model=self.model_name, input=[query])
        return response["embeddings"]