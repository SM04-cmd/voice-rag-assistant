import requests
from bs4 import BeautifulSoup
import sys
sys.path.append(".")
from rag.query import query_docs

# Tool 1 — Fetch full page content
def fetch_page(url: str) -> str:
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        content = soup.get_text(separator=' ', strip=True)
        return content[:3000]
    except Exception as e:
        return f"Error fetching page: {e}"

# Tool 2 — Search documentation
def search_docs(question: str) -> str:
    try:
        docs, urls = query_docs(question)
        result = ""
        for doc, url in zip(docs, urls):
            result += f"\nSource: {url}\n{doc[:500]}\n"
        return result
    except Exception as e:
        return f"Error searching docs: {e}"

if __name__ == "__main__":
    # Test tool 1
    print("Testing fetch_page tool...")
    content = fetch_page("https://fastapi.tiangolo.com")
    print(content[:200])
    
    # Test tool 2
    print("\nTesting search_docs tool...")
    result = search_docs("How to install FastAPI?")
    print(result[:200])