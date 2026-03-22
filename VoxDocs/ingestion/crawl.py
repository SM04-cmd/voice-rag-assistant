import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import urljoin

BASE_URL = "https://fastapi.tiangolo.com"
visited = set()
pages = []

def crawl(url, depth=0):
    if depth > 2 or url in visited:
        return
    visited.add(url)
    try:
        res = requests.get(url, timeout=5)
        soup = BeautifulSoup(res.text, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        pages.append({"url": url, "text": text})
        print(f"Crawled: {url}")
        for a in soup.find_all("a", href=True):
            link = urljoin(BASE_URL, a["href"])
            if link.startswith(BASE_URL) and link not in visited:
                crawl(link, depth + 1)
    except Exception as e:
        print(f"Failed {url}: {e}")

crawl(BASE_URL)
with open("ingestion/pages.json", "w") as f:
    json.dump(pages, f)
print(f"Done. {len(pages)} pages saved.")
