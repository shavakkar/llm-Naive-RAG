import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from ai.config import Config
from ai.ingestion.loader import LocalFileLoader
from ai.ingestion.chunker import RecursiveCharacterChunker
from ai.rag.chroma_client import ChromaVectorStore
from ai.rag.pipeline import RAGPipelineFactory

# 1. Load files from disk
loader = LocalFileLoader(docs_dir="./docs")
raw_docs = loader.load_documents()

# 2. Chunk documents recursively
chunker = RecursiveCharacterChunker(chunk_size=200, chunk_overlap=30)
chunked_docs = chunker.chunk_documents(raw_docs)

# 3. Embed documents
embedder = RAGPipelineFactory.create_embedder()
embeddings = embedder.embed_documents([doc["content"] for doc in chunked_docs])

# 4. Store in ChromaDB
vector_store = ChromaVectorStore(
    persist_dir=Config.CHROMA_PERSIST_DIR,
    collection_name=Config.COLLECTION_NAME
)
vector_store.add_documents(documents=chunked_docs, embeddings=embeddings)

# 5. Search test
query = "What authentication is required?"
query_vec = embedder.embed_query(query)
search_results = vector_store.query(query_vec, top_k=2)

print("\n--- RETRIEVAL RESULTS WITH METADATA & DISTANCE ---")
for res in search_results:
    print(res)