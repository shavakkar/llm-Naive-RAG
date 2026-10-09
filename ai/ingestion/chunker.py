from typing import List, Dict, Any

class RecursiveCharacterChunker:
    """
    Splits text recursively using structural separators (\n\n, \n, space, empty) 
    to preserve complete sentences and semantic meaning.
    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        if chunk_overlap >= chunk_size:
            raise ValueError("Overlap must be smaller than chunk size.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", ". ", " ", ""]

    def _split_text_with_separator(self, text: str, separator: str) -> List[str]:
        if separator == "":
            return list(text)
        return text.split(separator)

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        chunks = []
        current_chunk = []
        current_length = 0

        for split in splits:
            split_len = len(split)
            
            if current_length + split_len + (len(separator) if current_chunk else 0) <= self.chunk_size:
                current_chunk.append(split)
                current_length += split_len + (len(separator) if len(current_chunk) > 1 else 0)
            else:
                if current_chunk:
                    chunks.append(separator.join(current_chunk))
                current_chunk = [split]
                current_length = split_len

        if current_chunk:
            chunks.append(separator.join(current_chunk))

        return chunks

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Takes loaded document objects (from LocalFileLoader) and returns chunked objects:
        [
            {
                "content": "Chunk text paragraph...",
                "metadata": {"source": "sample_policy.txt", "chunk_id": 0}
            }
        ]
        """
        chunked_docs = []

        for doc in documents:
            text = doc["content"]
            metadata = doc["metadata"]
            
            # Simple recursive split
            raw_chunks = self._recursive_split(text, self.separators)
            
            for idx, chunk_text in enumerate(raw_chunks):
                if chunk_text.strip():
                    chunk_meta = metadata.copy()
                    chunk_meta["chunk_id"] = idx
                    chunked_docs.append({
                        "content": chunk_text.strip(),
                        "metadata": chunk_meta
                    })

        print(f"[RecursiveChunker] Generated {len(chunked_docs)} chunk(s) from {len(documents)} document(s).")
        return chunked_docs

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self.chunk_size or not separators:
            return [text]

        separator = separators[0]
        next_separators = separators[1:]
        splits = self._split_text_with_separator(text, separator)

        final_chunks = []
        good_splits = []

        for split in splits:
            if len(split) <= self.chunk_size:
                good_splits.append(split)
            else:
                if good_splits:
                    final_chunks.extend(self._merge_splits(good_splits, separator))
                    good_splits = []
                final_chunks.extend(self._recursive_split(split, next_separators))

        if good_splits:
            final_chunks.extend(self._merge_splits(good_splits, separator))

        return final_chunks