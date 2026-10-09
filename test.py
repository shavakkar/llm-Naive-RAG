from ai.ingestion.loader import LocalFileLoader
from ai.ingestion.chunker import RecursiveCharacterChunker

# 1. Load files from disk
loader = LocalFileLoader(docs_dir="./docs")
raw_docs = loader.load_documents()

# 2. Chunk documents recursively
chunker = RecursiveCharacterChunker(chunk_size=150, chunk_overlap=20)
chunked_docs = chunker.chunk_documents(raw_docs)

print("\n--- CHUNKED OUTPUT ---")
for chunk in chunked_docs:
    print(chunk)