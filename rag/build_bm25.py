import json
import pickle
from rank_bm25 import BM25Okapi

with open("ingestion/pages.json", "r", encoding="utf-8") as f:
    pages = json.load(f)

chunks = []
for page in pages:
    content = page.get("content") or page.get("text") or str(page)
    chunks.append(content)

tokenized = [chunk.lower().split() for chunk in chunks]
bm25 = BM25Okapi(tokenized)

with open("rag/bm25_index.pkl", "wb") as f:
    pickle.dump({"bm25": bm25, "chunks": chunks}, f)

print("✅ BM25 index created!")