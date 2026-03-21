# ingestion/ingest.py
import json
import os
import re
import pickle
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import chromadb
from dotenv import load_dotenv

load_dotenv()

MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def smart_chunk(text, chunk_size=500, overlap=50):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current, length = [], [], 0
    for sentence in sentences:
        words = sentence.split()
        if length + len(words) > chunk_size and current:
            chunks.append(" ".join(current))
            current = current[-overlap:] + words
            length = sum(len(w) for w in current)
        else:
            current.extend(words)
            length += len(words)
    if current:
        chunks.append(" ".join(current))
    return chunks

def run_ingestion():
    # Load crawled data
    data_path = "ingestion/pages.json"
    if not os.path.exists(data_path):
        print("ERROR: crawled_data.json not found! Run crawl.py first.")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        pages = json.load(f)
    print(f"Loaded {len(pages)} crawled pages")

    # Chunk all pages
    all_chunks, all_urls, all_titles, all_ids = [], [], [], []
    for i, page in enumerate(pages):
        chunks = smart_chunk(page.get("text", ""))
        for j, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_urls.append(page.get("url", ""))
            all_titles.append(page.get("title", ""))
            all_ids.append(f"doc_{i}_{j}")

    print(f"Total chunks created: {len(all_chunks)}")
    print("Generating embeddings... (this takes 2-3 minutes)")

    embeddings = MODEL.encode(all_chunks, show_progress_bar=True).tolist()

    # Store in Chroma
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Delete existing collection if it exists
    try:
        client.delete_collection("docs")
    except:
        pass
    
    collection = client.create_collection("docs")

    # Add in batches of 100
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        batch_end = min(i + batch_size, len(all_chunks))
        collection.add(
            documents=all_chunks[i:batch_end],
            embeddings=embeddings[i:batch_end],
            metadatas=[{"url": all_urls[k], "title": all_titles[k]} 
                      for k in range(i, batch_end)],
            ids=all_ids[i:batch_end]
        )
        print(f"Stored chunks {i} to {batch_end}")

    print("Saved to Chroma database!")

    # Save BM25 index
    tokenized = [c.split() for c in all_chunks]
    bm25 = BM25Okapi(tokenized)
    os.makedirs("rag", exist_ok=True)
    with open("rag/bm25_index.pkl", "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "chunks": all_chunks,
            "urls": all_urls
        }, f)
    print("BM25 index saved!")
    print("Ingestion complete!")

if __name__ == "__main__":
    run_ingestion()