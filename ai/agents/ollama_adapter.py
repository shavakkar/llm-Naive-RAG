import re
from typing import List, Dict, Any
import ollama
from ai.agents.base import BaseGenerator

class OllamaGenerator(BaseGenerator):
    """
    Ollama API Client Execution with graceful model detection.
    """
    def __init__(self, model_name: str | None, base_url: str):
        self.client = ollama.Client(host=base_url)
        self.model_name = self._resolve_model(model_name)
        print(f"[Generator] Connected to Ollama LLM Service using model: '{self.model_name}'")

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
                "   Please install an LLM by running:\n"
                "   👉 ollama pull qwen2.5:7b\n or"
                "   👉 ollama pull qwen3:8b\n"
                "   (or run: `ollama pull deepseek-r1:1.5b`)"
            )

        # 1. If user specified a model, check if it's installed
        if specified_model:
            if any(specified_model in m for m in installed_models):
                return specified_model
            print(f"⚠️ Specified LLM model '{specified_model}' not found in installed models.")

        # 2. Look for common LLM keywords among installed models (excluding dedicated embedding models)
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

        # Parse <think> tags if present in reasoning models like DeepSeek-R1
        think_match = re.search(r'<think>(.*?)</think>', raw_output, flags=re.DOTALL)
        thoughts = think_match.group(1).strip() if think_match else "No explicit reasoning generated."
        clean_answer = re.sub(r'<think>.*?</think>', '', raw_output, flags=re.DOTALL).strip()

        return {
            "thoughts": thoughts,
            "answer": clean_answer
        }