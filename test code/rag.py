# Install dependencies:
# pip install sentence-transformers chromadb transformers accelerate torch

import uuid
import chromadb
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==========================================================
# 0. Load Local Qwen3 Embedding Model
# ==========================================================

# Point to your local downloaded Qwen3 directory
QWEN_EMBED_PATH = r"C:\Softwares\LLMS\Qwen3-embed"

print("Loading Qwen3 embedding model...")

embed_model = SentenceTransformer(
    QWEN_EMBED_PATH,
    trust_remote_code=True,
    model_kwargs={"dtype": torch.float16}
)

def embed_text_local(texts, is_query=False):
    """
    Generate dense embeddings locally using Qwen3-Embedding-0.6B
    """
    # Qwen3 retrieval benefits from using prompt_name="query" for search queries
    kwargs = {"prompt_name": "query"} if is_query else {}
    
    embeddings = embed_model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=12,
        **kwargs
    )
    return embeddings.tolist()

# ==========================================================
# 1. Load Local LLM
# ==========================================================

LLM_PATH = r"C:\Softwares\LLMS\deepseek-r1-distill-qwen-1.5b"

print("Loading LLM...")

llm_tokenizer = AutoTokenizer.from_pretrained(LLM_PATH)

llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_PATH,
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
    "The Qwen3 model supports dense, sparse, and multi-vector representations.",
    "Qwen is a large language model developed for multilingual tasks."
]

# Generate dense embeddings for documents
embeddings = embed_text_local(documents, is_query=False)

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

    # Pass is_query=True to apply Qwen3's recommended query prompt structure
    query_embedding = embed_text_local([query], is_query=True)

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