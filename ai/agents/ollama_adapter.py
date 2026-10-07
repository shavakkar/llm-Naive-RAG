import re
from typing import List, Dict, Any
import ollama
from ai.agents.base import BaseGenerator

class OllamaGenerator(BaseGenerator):
    """
    Ollama API Client Execution.
    """
    def __init__(self, model_name: str, base_url: str):
        print(f"[Generator] Connecting to Ollama LLM Service ({model_name})...")
        self.model_name = model_name
        self.client = ollama.Client(host=base_url)

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

        # Parse <think> tags if present in models like DeepSeek-R1 via Ollama
        think_match = re.search(r'<think>(.*?)</think>', raw_output, flags=re.DOTALL)
        thoughts = think_match.group(1).strip() if think_match else "No explicit reasoning generated."
        clean_answer = re.sub(r'<think>.*?</think>', '', raw_output, flags=re.DOTALL).strip()

        return {
            "thoughts": thoughts,
            "answer": clean_answer
        }