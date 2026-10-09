import sys
import shutil
import re
from typing import List, Dict, Any
import ollama
from ai.agents.base import BaseGenerator

class OllamaGenerator(BaseGenerator):
    """
    Ollama API Client Execution with bulletproof model resolution via metadata inspection.
    """
    def __init__(self, model_name: str | None, base_url: str):
        self.client = ollama.Client(host=base_url)
        self.model_name = self._resolve_model(model_name)
        print(f"[Generator] Connected to Ollama LLM Service using model: '{self.model_name}'")

    def _verify_ollama_installation(self, error_details: Exception):
        """Checks if Ollama executable exists on system PATH and exits with a friendly message."""
        is_installed = shutil.which("ollama") is not None
        
        print("\n" + "=" * 60)
        if not is_installed:
            print("❌ ERROR: Ollama is not installed on your system.")
            print("👉 Please download and install Ollama from: https://ollama.com/download")
        else:
            print(f"❌ ERROR: Ollama is installed but the server is not running at '{self.client._host}'.")
            print("👉 Please start the Ollama service by running in your terminal:\n   ollama serve")
        print("=" * 60 + "\n")
        sys.exit(1)

    def _is_embedding_only_model(self, model_name: str) -> bool:
        """
        Queries Ollama API metadata to inspect model architecture parameters.
        """
        try:
            info = self.client.show(model_name)
            details = info.get("details", {})
            family = details.get("family", "").lower()
            
            # Check GGUF model families dedicated solely to embeddings
            if family in ["bert", "nomic-bert", "xlm-roberta", "colbert"]:
                return True
                
            modelfile = info.get("modelfile", "").lower()
            if "embedding_only" in modelfile or "embedding" in family:
                return True

            return False
        except Exception:
            # Fallback to string check if API metadata inspect fails
            return any(kw in model_name.lower() for kw in ["embed", "bge", "minilm"])

    def _resolve_model(self, specified_model: str | None) -> str:
        try:
            installed_response = self.client.list()
            installed_models = [
                m.get("model") if isinstance(m, dict) else getattr(m, "model", getattr(m, "name", str(m)))
                for m in installed_response.get("models", [])
            ]
        except Exception as e:
            self._verify_ollama_installation(e)

        # Categorize installed models using architecture metadata
        llm_candidates = []
        embed_models = []

        for model in installed_models:
            if self._is_embedding_only_model(model):
                embed_models.append(model)
            else:
                llm_candidates.append(model)

        # 1. Validate specified model
        if specified_model:
            if any(specified_model in m for m in installed_models):
                if specified_model in embed_models or self._is_embedding_only_model(specified_model):
                    print("\n" + "=" * 60)
                    print(f"❌ ERROR: '{specified_model}' is an embedding-only model and cannot generate text.")
                    print("👉 Please specify or pull a generative LLM (e.g., `ollama pull qwen2.5:7b`).")
                    print("=" * 60 + "\n")
                    sys.exit(1)
                return specified_model

        # 2. Select first valid LLM candidate
        if llm_candidates:
            selected = llm_candidates[0]
            if specified_model:
                print(f"⚠️ Specified model '{specified_model}' not found. Auto-selecting LLM: '{selected}'")
            return selected

        # 3. Handle zero-LLM edge case cleanly
        print("\n" + "=" * 60)
        print("❌ ERROR: No text generation LLM models found in your local Ollama instance.")
        if embed_models:
            print(f"   (Detected embedding-only models: {embed_models}).")
        print("👉 Please download a text generation LLM by running:")
        print("   ollama pull qwen2.5:7b")
        print("=" * 60 + "\n")
        sys.exit(1)

    def generate_answer(self, query: str, context_documents: List[str]) -> Dict[str, Any]:
        context = "\n---\n".join(context_documents)
        prompt = f"""You are a helpful assistant. Use ONLY the provided context to answer.

Context:
{context}

Question:
{query}

Answer:"""

        response = self.client.generate(model=self.model_name, prompt=prompt)
        raw_output = response["response"]

        think_match = re.search(r'<think>(.*?)</think>', raw_output, flags=re.DOTALL)
        thoughts = think_match.group(1).strip() if think_match else "No explicit reasoning generated."
        clean_answer = re.sub(r'<think>.*?</think>', '', raw_output, flags=re.DOTALL).strip()

        return {
            "thoughts": thoughts,
            "answer": clean_answer
        }