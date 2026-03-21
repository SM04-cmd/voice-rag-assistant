import json
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# Load crawled pages
with open("ingestion/pages.json", "r", encoding="utf-8") as f:
    pages = json.load(f)

print(f"📄 Loaded {len(pages)} pages")

# Setup ChromaDB
client = chromadb.PersistentClient(path="./rag/vectordb")
collection = client.get_or_create_collection(name="fastapi_docs")

# Load embedding model
print("🔄 Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Add pages to vector DB
print("🚀 Adding pages to vector DB...")
for i, page in enumerate(pages):
    content = page.get("content") or page.get("text") or str(page)
    url = page.get("url", "unknown")
    
    # Generate embedding
    embedding = model.encode(content).tolist()
    
    collection.add(
        documents=[content],
        embeddings=[embedding],
        metadatas=[{"url": url}],
        ids=[f"page_{i}"]
    )
    print(f"✅ Added page {i+1}/{len(pages)}: {url}")

print("🎉 Done! Vector DB is ready!")