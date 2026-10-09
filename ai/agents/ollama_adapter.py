import sys
import shutil
import re
from typing import List, Dict, Any
import ollama
from ai.agents.base import BaseGenerator

class OllamaGenerator(BaseGenerator):
    """
    Ollama API Client Execution with system-level installation & model checks.
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
            print("❌ ERROR: No LLM models found in your local Ollama instance.")
            print("👉 Please download an LLM by running:\n")
            print("   ollama pull qwen2.5:7b\n")
            print("   ollama pull qwen3:8b\n")
            print("=" * 60 + "\n")
            sys.exit(1)

        # 1. Match specified model
        if specified_model:
            if any(specified_model in m for m in installed_models):
                return specified_model
            print(f"⚠️ Specified LLM model '{specified_model}' not found in installed models.")

        # 2. Look for LLM keyword match
        llm_keywords = ["qwen", "llama", "deepseek", "mistral", "gemma", "phi"]
        for model in installed_models:
            if any(kw in model.lower() for kw in llm_keywords) and "embed" not in model.lower():
                return model

        # 3. Fallback to first available model
        fallback = installed_models[0]
        print(f"ℹ️ Falling back to installed model: '{fallback}'")
        return fallback

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