import os
from enum import Enum
from pathlib import Path
import ollama

# Workspace Root Directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

class ExecutionProvider(str, Enum):
    LOCAL = "local"      # Local PyTorch + Hugging Face / SentenceTransformers
    OLLAMA = "ollama"    # Ollama REST Client API

def get_installed_ollama_models(base_url: str) -> list[str]:
    """Helper to safely fetch installed model names from Ollama."""
    try:
        client = ollama.Client(host=base_url)
        response = client.list()
        # ollama python client returns objects with .model or dicts with 'name' / 'model'
        models = []
        for item in response.get("models", []):
            if isinstance(item, dict):
                models.append(item.get("model") or item.get("name"))
            else:
                models.append(getattr(item, "model", getattr(item, "name", str(item))))
        return [m for m in models if m]
    except Exception:
        return []

class Config:
    # --- GLOBAL PROVIDER TOGGLE ---
    PROVIDER: ExecutionProvider = ExecutionProvider.OLLAMA

    # --- LOCAL TRANSFORMERS / PYTORCH PATHS ---
    LOCAL_EMBED_PATH: str = r"C:\Softwares\LLMS\Qwen3-embed"
    LOCAL_LLM_PATH: str = r"C:\Softwares\LLMS\deepseek-r1-distill-qwen-1.5b"

    # --- OLLAMA SERVICE CONFIGURATION ---
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    
    # Optional explicitly defined models (leave as None to auto-detect from user's installed models)
    OLLAMA_EMBED_MODEL: str | None = None
    OLLAMA_LLM_MODEL: str | None = None

    # --- VECTOR DATABASE CONFIGURATION ---
    CHROMA_PERSIST_DIR: str = str(BASE_DIR / "infrastructure" / "chroma" / "chroma_store")
    COLLECTION_NAME: str = "enterprise_docs"

    # --- RAG PARAMETERS ---
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K_RESULTS: int = 2