from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseGenerator(ABC):
    """
    Abstract Interface for LLM Reasoning and Generation.
    """
    @abstractmethod
    def generate_answer(self, query: str, context_documents: List[str]) -> Dict[str, Any]:
        """
        Processes query and context, returning a dict containing:
        {
            "thoughts": str,
            "answer": str
        }
        """
        pass