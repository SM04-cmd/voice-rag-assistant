# ingestion/crawl.py
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
import os
import time

BASE_URL = "https://fastapi.tiangolo.com"

def crawl(base_url, max_pages=50):
    visited = set()
    pages = []
    queue = [base_url]

    while queue and len(pages) < max_pages:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        try:
            r = requests.get(url, timeout=10)
            soup = BeautifulSoup(r.text, "html.parser")
            content = soup.find("article") or soup.find("main") or soup.body
            text = content.get_text(separator="\n", strip=True) if content else ""

            if len(text) > 200:
                pages.append({
                    "url": url,
                    "title": soup.title.string if soup.title else url,
                    "text": text
                })
                print(f"✅ Crawled ({len(pages)}): {url}")

            for a in soup.find_all("a", href=True):
                full_url = urljoin(url, a["href"])
                if urlparse(full_url).netloc == urlparse(base_url).netloc:
                    if full_url not in visited:
                        queue.append(full_url)

            time.sleep(0.3)

        except Exception as e:
            print(f"❌ Error: {url} — {e}")

    return pages

if __name__ == "__main__":
    print("🚀 Starting crawl...")
    pages = crawl(BASE_URL)
    
    os.makedirs("ingestion", exist_ok=True)
    
    save_path = "ingestion/pages.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(pages, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Done! Crawled {len(pages)} pages")
    print(f"💾 Saved to {save_path}")