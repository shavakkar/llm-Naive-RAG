# Install dependencies:
# pip install chromadb nomic transformers accelerate torch

import chromadb
from nomic import embed
import uuid
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ---------------------------
# 1. Load your local Qwen model
# ---------------------------
# Replace with your local path from Kaggle download
MODEL_PATH = "C:\Softwares\LLMS\deepseek-r1-distill-qwen-1.5b"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)

# ---------------------------
# 2. Initialize ChromaDB
# ---------------------------
client = chromadb.PersistentClient(path="./chroma_store")
collection = client.create_collection(name="my_docs")

# ---------------------------
# 3. Add documents to ChromaDB
# ---------------------------
documents = [
    "Python is a popular programming language for AI and data science.",
    "ChromaDB is a vector database for storing and retrieving embeddings.",
    "The nomic-embed-text model generates embeddings for text.",
    "Qwen is a large language model developed for multilingual tasks."
]

embeddings = embed.text(
    texts=documents,
    model="nomic-embed-text-v1",
    task_type="search_document"
)["embeddings"]

collection.add(
    ids=[str(uuid.uuid4()) for _ in documents],
    documents=documents,
    embeddings=embeddings
)

print("✅ Documents indexed in ChromaDB.")

# ---------------------------
# 4. Retrieval + Generation
# ---------------------------
def retrieve(query, top_k=2):
    query_embedding = embed.text(
        texts=[query],
        model="nomic-embed-text-v1.5",
        task_type="search_query"
    )["embeddings"]

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    return results["documents"][0]

def generate_answer(query):
    # Retrieve relevant docs
    docs = retrieve(query)
    context = "\n".join(docs)

    # Build prompt for Qwen
    prompt = f"""You are a helpful assistant.
Use the following context to answer the question.

Context:
{context}

Question: {query}
Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ---------------------------
# 5. Test the RAG pipeline
# ---------------------------
user_query = "What is ChromaDB?"
answer = generate_answer(user_query)

print("\n🔍 Query:", user_query)
print("💡 Answer:\n", answer)