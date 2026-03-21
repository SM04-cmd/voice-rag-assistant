# VoxDocs 🎙️
> Voice AI Documentation Assistant powered by RAG + Groq + LiveKit

## Architecture
```
User Voice → LiveKit Agent → STT (Groq Whisper) → RAG Query → Vector DB (ChromaDB) → Groq LLM → TTS → Voice Response
```

## RAG Optimizations
1. **Hybrid Retrieval** — Combines vector search (ChromaDB) + BM25 keyword search for better results
2. **Context Limiting** — Prevents token overflow and reduces latency

## Tools
1. **fetch_page(url)** — Fetches full documentation page when deeper context is needed
2. **search_docs(question)** — Searches vector database for relevant documentation chunks

## Tech Stack
- 🧠 LLM: Groq (llama-3.3-70b)
- 🎤 STT: Groq Whisper
- 📦 Vector DB: ChromaDB
- 🔍 Search: BM25 + Vector Hybrid
- 🎙️ Voice: LiveKit

## Setup Instructions

### 1. Clone the repo
```bash
git clone https://github.com/SM04-cmd/voice-rag-assistant
cd voice-rag-assistant
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Create .env file
```
LIVEKIT_URL=your_livekit_url
LIVEKIT_API_KEY=your_api_key
LIVEKIT_API_SECRET=your_api_secret
GROQ_API_KEY=your_groq_key
```

### 4. Crawl documentation
```bash
py ingestion/crawl.py
```

### 5. Build RAG index
```bash
py rag/ingest.py
py rag/build_bm25.py
```

### 6. Run voice agent
```bash
py voice_agent/agent.py
```

## Project Structure
```
VoxDocs/
├── ingestion/
│   ├── crawl.py       # Web crawler
│   └── pages.json     # Crawled pages
├── rag/
│   ├── ingest.py      # Vector DB builder
│   ├── query.py       # RAG query engine
│   └── build_bm25.py  # BM25 index builder
├── tools/
│   └── tools.py       # Agent tools
├── voice_agent/
│   └── agent.py       # Main voice agent
└── .env               # API keys (not committed)
```