import sys
sys.path.append(".")

import os
import pickle
import chromadb
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, Agent, AgentSession
from livekit.plugins import groq as livekit_groq
from livekit.plugins import silero

load_dotenv()

# Load models
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# Load ChromaDB
chroma_client = chromadb.PersistentClient(path="./rag/vectordb")
collection = chroma_client.get_collection("fastapi_docs")

# Load BM25
with open("rag/bm25_index.pkl", "rb") as f:
    bm25_data = pickle.load(f)

def retrieve(query, top_k=2):
    embedding = EMBED_MODEL.encode([query]).tolist()
    results = collection.query(query_embeddings=embedding, n_results=5)
    vector_chunks = results["documents"][0]

    tokens = query.lower().split()
    scores = bm25_data["bm25"].get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:5]
    bm25_chunks = [bm25_data["chunks"][i] for i in top_indices if scores[i] > 0]

    seen = set()
    combined = []
    for chunk in vector_chunks + bm25_chunks:
        if chunk not in seen:
            seen.add(chunk)
            combined.append(chunk)
    return combined[:top_k]


class VoxDocsAgent(Agent):
    def __init__(self):
        # Build system prompt with RAG context injected at runtime
        super().__init__(
            instructions="""You are VoxDocs, a helpful voice assistant for documentation.
Answer questions clearly and concisely based on the provided documentation context.
Keep responses short and conversational since you are speaking out loud."""
        )

    async def on_user_turn_completed(self, turn_ctx, new_message):
        # Inject RAG context into every user message
        user_text = new_message.text_content
        chunks = retrieve(user_text, top_k=2)
        context = "\n\n".join(chunks)

        new_message.content = (
            f"Documentation context:\n{context}\n\n"
            f"User question: {user_text}"
        )
        await super().on_user_turn_completed(turn_ctx, new_message)


async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    session = AgentSession(
        stt=livekit_groq.STT(model="whisper-large-v3"),        # Speech-to-text
        llm=livekit_groq.LLM(model="llama-3.3-70b-versatile"), # LLM
        tts=livekit_groq.TTS(model="playai-tts",               # ✅ Groq TTS
                              voice="Celeste-PlayAI"),
        vad=silero.VAD.load(),                                  # Voice activity detection
    )

    await session.start(
        agent=VoxDocsAgent(),
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))