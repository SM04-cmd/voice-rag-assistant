import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# Setup ChromaDB
client = chromadb.PersistentClient(path="./rag/vectordb")
collection = client.get_or_create_collection(name="fastapi_docs")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def query_docs(question: str, n_results: int = 3):
    # Generate embedding for question
    embedding = model.encode(question).tolist()
    
    # Search vector DB
    results = collection.query(
        query_embeddings=[embedding],
        n_results=n_results
    )
    
    docs = results["documents"][0]
    urls = [m["url"] for m in results["metadatas"][0]]
    
    return docs, urls

if __name__ == "__main__":
    question = "How do I create a FastAPI app?"
    docs, urls = query_docs(question)
    
    print(f"🔍 Question: {question}")
    print(f"\n📄 Top results:")
    for i, (doc, url) in enumerate(zip(docs, urls)):
        print(f"\n{i+1}. {url}")
        print(doc[:200])