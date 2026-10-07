# Install dependencies:
# pip install FlagEmbedding chromadb transformers accelerate torch

import uuid
import chromadb
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

# ==========================================================
# 0. Load Qwen3 Embedding Model (FlagEmbedding)
# ==========================================================

print("Loading Qwen3 embedding model...")

# use_fp16=True uses FP16 for faster computation on CUDA GPUs
embed_model = BGEM3FlagModel("BAAI/Qwen3", use_fp16=True)

def embed_text_local(texts):
    """
    Generate dense embeddings locally using Qwen3 via FlagEmbedding
    """
    # encode returns a dict containing 'dense_vecs', 'lexical_weights', and 'colbert_vecs'
    output = embed_model.encode(
        texts,
        batch_size=12,
        max_length=8192  # Qwen3 supports long contexts up to 8192 tokens
    )
    return output["dense_vecs"].tolist()

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
    "The Qwen3 model supports dense, sparse, and multi-vector representations.",
    "Qwen is a large language model developed for multilingual tasks."
]

# Generate dense embeddings directly without prefixed prompts
embeddings = embed_text_local(documents)

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

    query_embedding = embed_text_local([query])

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