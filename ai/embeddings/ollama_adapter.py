import sys
import shutil
from typing import List
import ollama
from ai.embeddings.base import BaseEmbedder

class OllamaEmbedder(BaseEmbedder):
    """
    Ollama REST Client Execution with system-level installation & model checks.
    """
    def __init__(self, model_name: str | None, base_url: str):
        self.client = ollama.Client(host=base_url)
        self.model_name = self._resolve_model(model_name)
        print(f"[Embedder] Connected to Ollama Embedding Service using model: '{self.model_name}'")

    def _verify_ollama_installation(self, error_details: Exception):
        """Checks if Ollama executable exists on system PATH and exits with a friendly message."""
        is_installed = shutil.which("ollama") is not None
        
        print("\n" + "=" * 60)
        if not is_installed:
            print("❌ ERROR: Ollama is not installed on your system.")
            print("👉 Please download and install Ollama from: https://ollama.com/download")
        else:
            print(f"❌ ERROR: Ollama is installed but the server is not running at '{self.client._host}'.")
            print("👉 Please start the Ollama service by running in your terminal:")
            print("   ollama serve")
        print("=" * 60 + "\n")
        sys.exit(1)

    def _resolve_model(self, specified_model: str | None) -> str:
        try:
            installed_response = self.client.list()
            installed_models = [
                m.get("model") if isinstance(m, dict) else getattr(m, "model", getattr(m, "name", str(m)))
                for m in installed_response.get("models", [])
            ]
        except Exception as e:
            self._verify_ollama_installation(e)

        if not installed_models:
            print("\n" + "=" * 60)
            print("❌ ERROR: No models found in your local Ollama instance.")
            print("👉 Please download an embedding model by running:")
            print("   ollama pull nomic-embed-text")
            print("=" * 60 + "\n")
            sys.exit(1)

        # 1. Match specified model
        if specified_model:
            if any(specified_model in m for m in installed_models):
                return specified_model
            print(f"⚠️ Specified embed model '{specified_model}' not found in installed models.")

        # 2. Look for embedding keyword match
        embedding_keywords = ["embed", "bge", "nomic", "minilm"]
        for model in installed_models:
            if any(kw in model.lower() for kw in embedding_keywords):
                return model

        # 3. Fallback to first available model
        fallback = installed_models[0]
        print(f"ℹ️ No dedicated embedding model detected. Falling back to installed model: '{fallback}'")
        return fallback

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        response = self.client.embed(model=self.model_name, input=documents)
        return response["embeddings"]

    def embed_query(self, query: str) -> List[List[float]]:
        response = self.client.embed(model=self.model_name, input=[query])
        return response["embeddings"]