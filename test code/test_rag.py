import sys
from pathlib import Path

# Append root directory to sys.path to enable workspace imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ai.config import Config, ExecutionProvider
from ai.rag.chroma_client import ChromaVectorStore
from ai.rag.pipeline import RAGPipelineFactory, RAGPipeline

def main():
    print(f"=== Starting RAG Execution Pipeline (Provider Mode: {Config.PROVIDER.value.upper()}) ===")

    # 1. Instantiate Adapters via Factory
    embedder = RAGPipelineFactory.create_embedder()
    generator = RAGPipelineFactory.create_generator()

    # 2. Initialize Vector Store
    vector_store = ChromaVectorStore(
        persist_dir=Config.CHROMA_PERSIST_DIR,
        collection_name=Config.COLLECTION_NAME
    )

    # 3. Assemble Pipeline
    pipeline = RAGPipeline(
        embedder=embedder,
        generator=generator,
        vector_store=vector_store
    )

    # 4. Ingest Corpus
    documents = [
        "Python is a popular programming language for AI and data science.",
        "ChromaDB is an open-source vector database for storing and retrieving embeddings.",
        "The Qwen3 model supports dense, sparse, and multi-vector representations.",
        "Qwen is a large language model family developed for multilingual AI tasks."
    ]
    pipeline.ingest_documents(documents)

    # 5. Run Search & Generation Test
    user_query = "What is ChromaDB?"
    output = pipeline.query(user_query)

    print("\n" + "="*50)
    print("🔍 QUERY:", user_query)
    print("="*50)
    print("\n📚 RETRIEVED CONTEXT:")
    for idx, doc in enumerate(output["retrieved_docs"], 1):
        print(f"  [{idx}] {doc}")
        
    print("\n🧠 INTERNAL LLM THOUGHTS (COGNITIVE PROCESS):")
    print(output["thoughts"])
    
    print("\n💡 FINAL RESPONSE:")
    print(output["answer"])
    print("="*50)

if __name__ == "__main__":
    main()