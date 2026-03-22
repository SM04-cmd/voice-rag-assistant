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
from livekit.plugins import cartesia

load_dotenv()

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="./rag/vectordb")
collection = chroma_client.get_collection("fastapi_docs")

with open("rag/bm25_index.pkl", "rb") as f:
    bm25_data = pickle.load(f)

def retrieve(query, top_k=1):
    embedding = EMBED_MODEL.encode([query]).tolist()
    results = collection.query(query_embeddings=embedding, n_results=2)
    vector_chunks = results["documents"][0]
    tokens = query.lower().split()
    scores = bm25_data["bm25"].get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:2]
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
        super().__init__(
            instructions="""You are VoxDocs, a helpful voice assistant for FastAPI documentation.
Keep ALL responses under 3 sentences. Be direct and conversational."""
        )

    async def on_user_turn_completed(self, turn_ctx, new_message):
        user_text = new_message.text_content
        chunks = retrieve(user_text, top_k=1)
        context = chunks[0][:300] if chunks else ""
        new_message.content = (
            f"Context: {context}\n\nQuestion: {user_text}"
        )
        await super().on_user_turn_completed(turn_ctx, new_message)


async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    session = AgentSession(
        stt=livekit_groq.STT(model="whisper-large-v3"),
        llm=livekit_groq.LLM(model="llama-3.3-70b-versatile"),
        tts=cartesia.TTS(),
        vad=silero.VAD.load(),
    )

    await session.start(
        agent=VoxDocsAgent(),
        room=ctx.room,
    )

    await session.generate_reply(
        instructions="Greet the user in one sentence. Say you are VoxDocs, ready to answer FastAPI questions."
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))