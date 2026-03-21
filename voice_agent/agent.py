# voice_agent/agent.py
import asyncio
import os
import pickle
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np
from groq import Groq

load_dotenv()

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load Chroma
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection("docs")

# Load BM25
with open("rag/bm25_index.pkl", "rb") as f:
    bm25_data = pickle.load(f)

def retrieve(query, top_k=4):
    # Vector search
    embedding = EMBED_MODEL.encode([query]).tolist()
    results = collection.query(query_embeddings=embedding, n_results=10)
    vector_chunks = results["documents"][0]

    # BM25 search
    tokens = query.lower().split()
    scores = bm25_data["bm25"].get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:10]
    bm25_chunks = [bm25_data["chunks"][i] for i in top_indices if scores[i] > 0]

    # Combine and deduplicate
    seen = set()
    combined = []
    for chunk in vector_chunks + bm25_chunks:
        if chunk not in seen:
            seen.add(chunk)
            combined.append(chunk)

    return combined[:top_k]

def ask_groq(question, context):
    prompt = f"""You are a helpful voice assistant that answers questions about FastAPI documentation.
Answer in simple, clear spoken language. Keep it under 3 sentences.

Context from documentation:
{context}

Question: {question}
Answer:"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )
    return response.choices[0].message.content

def main():
    print("🎙️ Voice RAG Assistant Ready!")
    print("Type your question (or 'quit' to exit):")
    print("-" * 40)

    while True:
        question = input("\nYou: ").strip()
        if question.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        if not question:
            continue

        print("Searching documentation...")
        chunks = retrieve(question)
        context = "\n\n".join(chunks)

        print("Generating answer...")
        answer = ask_groq(question, context)

        print(f"\nAssistant: {answer}")

if __name__ == "__main__":
    main()