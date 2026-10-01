# Install dependencies:
# pip install chromadb sentence-transformers transformers accelerate torch

import uuid
import chromadb
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==========================================================
# 0. Load Nomic Embedding Model
# ==========================================================

EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"

print("Loading embedding model...")

embed_model = SentenceTransformer(
    EMBED_MODEL,
    trust_remote_code=True
)

def embed_text_local(texts):
    """
    Generate embeddings locally using Nomic v1.5
    """
    return embed_model.encode(
        texts,
        normalize_embeddings=True
    ).tolist()

# ==========================================================
# 1. Load Local LLM
# ==========================================================

MODEL_PATH = r"C:\Softwares\LLMS\deepseek-r1-distill-qwen-1.5b"

print("Loading LLM...")

llm_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

llm_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)

# ==========================================================
# 2. Initialize ChromaDB
# ==========================================================

client = chromadb.PersistentClient(
    path="./chroma_store"
)

collection = client.get_or_create_collection(
    name="my_docs"
)

# ==========================================================
# 3. Sample Documents
# ==========================================================

documents = [
    "Python is a popular programming language for AI and data science.",
    "ChromaDB is a vector database for storing and retrieving embeddings.",
    "The nomic-embed-text model generates embeddings for text.",
    "Qwen is a large language model developed for multilingual tasks."
]

# Add search_document prefix (recommended for Nomic)

documents_for_embedding = [
    f"search_document: {doc}"
    for doc in documents
]

embeddings = embed_text_local(
    documents_for_embedding
)

# Optional: clear collection during testing
# collection.delete(where={})

collection.add(
    ids=[str(uuid.uuid4()) for _ in documents],
    documents=documents,
    embeddings=embeddings
)

print("✅ Documents indexed in ChromaDB.")

# ==========================================================
# 4. Retrieval
# ==========================================================

def retrieve(query, top_k=2):

    query_embedding = embed_text_local(
        [f"search_query: {query}"]
    )

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    return results["documents"][0]

# ==========================================================
# 5. Generation
# ==========================================================

def generate_answer(query):

    docs = retrieve(query)

    context = "\n".join(docs)

    prompt = f"""
You are a helpful assistant.

Use ONLY the provided context to answer.

Context:
{context}

Question:
{query}

Answer:
"""

    inputs = llm_tokenizer(
        prompt,
        return_tensors="pt"
    ).to(llm_model.device)

    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True
    )

    response = llm_tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return response

# ==========================================================
# 6. Test
# ==========================================================

user_query = "What is ChromaDB?"

answer = generate_answer(user_query)

print("\n🔍 Query:", user_query)
print("\n💡 Answer:\n")
print(answer)