import json
import pickle
from rank_bm25 import BM25Okapi

with open("ingestion/pages.json") as f:
    pages = json.load(f)

chunks = [p["text"][:2000] for p in pages]
tokenized = [c.lower().split() for c in chunks]
bm25 = BM25Okapi(tokenized)

with open("rag/bm25_index.pkl", "wb") as f:
    pickle.dump({"bm25": bm25, "chunks": chunks}, f)
print(f"Done. BM25 index built with {len(chunks)} chunks.")
