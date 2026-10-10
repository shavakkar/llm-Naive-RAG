import os
import sys
from enum import Enum
from pathlib import Path
import ollama

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class ExecutionProvider(str, Enum):
    LOCAL = "local"
    OLLAMA = "ollama"

def get_categorized_ollama_models(base_url: str = "http://localhost:11434") -> dict:
    """
    Queries Ollama API and categorizes installed models into 'embedders' and 'generators'.
    """
    try:
        client = ollama.Client(host=base_url)
        response = client.list()
        
        models = [
            m.get("model") if isinstance(m, dict) else getattr(m, "model", getattr(m, "name", str(m)))
            for m in response.get("models", [])
        ]
        
        embedders = []
        generators = []

        for model_name in models:
            try:
                info = client.show(model_name)
                family = info.get("details", {}).get("family", "").lower()
                modelfile = info.get("modelfile", "").lower()
                
                if family in ["bert", "nomic-bert", "xlm-roberta", "colbert"] or "embedding" in modelfile:
                    embedders.append(model_name)
                else:
                    generators.append(model_name)
            except Exception:
                if any(kw in model_name.lower() for kw in ["embed", "bge", "nomic", "minilm"]):
                    embedders.append(model_name)
                else:
                    generators.append(model_name)

        return {"embedders": embedders, "generators": generators}
    except Exception:
        return {"embedders": [], "generators": []}

class Config:
    # --- GLOBAL PROVIDER TOGGLE ---
    # Switch between ExecutionProvider.LOCAL and ExecutionProvider.OLLAMA
    PROVIDER: ExecutionProvider = ExecutionProvider(
        os.getenv("RAG_PROVIDER", ExecutionProvider.LOCAL.value).lower()
    )

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