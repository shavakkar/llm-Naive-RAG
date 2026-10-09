import os
from pathlib import Path
from typing import List, Dict, Any

class LocalFileLoader:
    """
    Scans and loads plain text (.txt) and Markdown (.md) documents from a local directory.
    """
    SUPPORTED_EXTENSIONS = {".txt", ".md"}

    def __init__(self, docs_dir: str):
        self.docs_dir = Path(docs_dir)
        if not self.docs_dir.exists():
            self.docs_dir.mkdir(parents=True, exist_ok=True)

    def load_documents(self) -> List[Dict[str, Any]]:
        """
        Reads all supported files in the target directory and returns a list of dictionaries:
        [
            {
                "content": "Raw file text...",
                "metadata": {"source": "filename.txt", "path": "/full/path/filename.txt"}
            }
        ]
        """
        documents = []
        
        for file_path in self.docs_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                        
                    if text:  # Ignore empty files
                        documents.append({
                            "content": text,
                            "metadata": {
                                "source": file_path.name,
                                "path": str(file_path.resolve())
                            }
                        })
                        print(f"[LocalFileLoader] Loaded: {file_path.name}")
                except Exception as e:
                    print(f"[LocalFileLoader] Error reading {file_path.name}: {e}")
                    
        print(f"[LocalFileLoader] Total documents loaded from disk: {len(documents)}")
        return documents