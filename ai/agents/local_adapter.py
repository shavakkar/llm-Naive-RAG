import re
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from ai.agents.base import BaseGenerator

class LocalDeepSeekGenerator(BaseGenerator):
    """
    Local PyTorch Execution for DeepSeek-R1 Distill Models.
    """
    def __init__(self, model_path: str):
        print(f"[Generator] Loading Local LLM from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def generate_answer(self, query: str, context_documents: List[str]) -> Dict[str, Any]:
        context = "\n---\n".join(context_documents)
        prompt = f"""You are a helpful assistant. Use ONLY the provided context to answer.

Context:
{context}

Question:
{query}

Answer:"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            do_sample=True
        )

        raw_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Parse <think> tags to capture model reasoning
        think_match = re.search(r'<think>(.*?)</think>', raw_output, flags=re.DOTALL)
        thoughts = think_match.group(1).strip() if think_match else "No explicit reasoning generated."
        clean_answer = re.sub(r'<think>.*?</think>', '', raw_output, flags=re.DOTALL).strip()

        return {
            "thoughts": thoughts,
            "answer": clean_answer
        }