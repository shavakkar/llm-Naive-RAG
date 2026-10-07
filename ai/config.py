import os
from enum import Enum
from pathlib import Path

# Workspace Root Directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

class ExecutionProvider(str, Enum):
    LOCAL = "local"      # Local PyTorch + Hugging Face / SentenceTransformers
    OLLAMA = "ollama"    # Ollama REST Client API

class Config:
    # --- GLOBAL PROVIDER TOGGLE ---
    # Switch between ExecutionProvider.LOCAL and ExecutionProvider.OLLAMA
    PROVIDER: ExecutionProvider = ExecutionProvider.LOCAL

    # --- LOCAL TRANSFORMERS / PYTORCH PATHS ---
    LOCAL_EMBED_PATH: str = r"C:\Softwares\LLMS\Qwen3-embed"
    LOCAL_LLM_PATH: str = r"C:\Softwares\LLMS\deepseek-r1-distill-qwen-1.5b"

    # --- OLLAMA SERVICE CONFIGURATION ---
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_EMBED_MODEL: str = "nomic-embed-text"  # e.g., nomic-embed-text, qwen3-embedding
    OLLAMA_LLM_MODEL: str = "qwen2.5:7b"          # e.g., qwen2.5:7b, deepseek-r1:1.5b

    # --- VECTOR DATABASE CONFIGURATION ---
    CHROMA_PERSIST_DIR: str = str(BASE_DIR / "infrastructure" / "chroma" / "chroma_store")
    COLLECTION_NAME: str = "enterprise_docs"

    # --- RAG PARAMETERS ---
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K_RESULTS: int = 2