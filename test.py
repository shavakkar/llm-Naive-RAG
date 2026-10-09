import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from ai.config import Config, ExecutionProvider
from ai.ingestion.loader import LocalFileLoader
from ai.ingestion.chunker import RecursiveCharacterChunker
from ai.rag.chroma_client import ChromaVectorStore
from ai.rag.pipeline import RAGPipelineFactory, RAGPipeline

def main():
    # Trigger model selection menu if running Ollama provider
    if Config.PROVIDER == ExecutionProvider.OLLAMA:
        RAGPipelineFactory.prompt_ollama_model_selection()

    # 1. Load files from disk
    loader = LocalFileLoader(docs_dir="./docs")
    raw_docs = loader.load_documents()

    # 2. Chunk documents recursively
    chunker = RecursiveCharacterChunker(chunk_size=300, chunk_overlap=40)
    chunked_docs = chunker.chunk_documents(raw_docs)

    # 3. Instantiate components via factory
    embedder = RAGPipelineFactory.create_embedder()
    generator = RAGPipelineFactory.create_generator()

    # 4. Initialize Vector Store
    vector_store = ChromaVectorStore(
        persist_dir=Config.CHROMA_PERSIST_DIR,
        collection_name=Config.COLLECTION_NAME
    )

    # 5. Assemble and execute pipeline
    pipeline = RAGPipeline(
        embedder=embedder,
        generator=generator,
        vector_store=vector_store
    )

    pipeline.ingest_documents(chunked_docs)

    # 6. Test Query
    user_query = "What is the access control policy?"
    output = pipeline.query(user_query)

    print("\n" + "="*50)
    print("🔍 QUERY:", user_query)
    print("="*50)
    print("\n💡 FINAL RESPONSE:")
    print(output["answer"])
    print("="*50)

if __name__ == "__main__":
    main()