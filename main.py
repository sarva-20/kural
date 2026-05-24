import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from thinnu.scraper import thinnu_eat
from brain.chunker import chunk_scraped_data
from brain.embedder import embed_chunks, find_relevant_chunks
from brain.ollama import (
    brain_think,
    brain_think_stream,
    brain_think_with_history,
    brain_think_stream_with_history,
    check_ollama_connection,
)

app = FastAPI(
    title="Kural",
    description="Thinnu eats the web. Brain understands it. You ask anything.",
    version="2.0.0",
)

# In-memory session store
# Session structure:
# {
#   "data": raw scraped data,
#   "embedded_chunks": list of embedded chunks,
#   "history": list of {"role": "user/assistant", "content": "..."}
# }
sessions: dict[str, dict] = {}


# ---------- Request Models ----------

class ScrapeRequest(BaseModel):
    url: str


class AskRequest(BaseModel):
    session_id: str
    question: str


# ---------- Routes ----------

@app.get("/")
def root():
    return {
        "name": "Kural 🔍",
        "tagline": "Thinnu eats. Brain thinks. You ask.",
        "version": "2.0.0",
        "architecture": "RAG (Retrieval Augmented Generation)",
        "models": {
            "chat": "mistral:7b",
            "embeddings": "nomic-embed-text",
        },
        "endpoints": {
            "POST /scrape": "Give Thinnu a URL. It eats, chunks, and embeds everything.",
            "POST /ask": "Ask the Brain anything. RAG retrieves, Mistral answers.",
            "POST /ask/stream": "Stream an answer from the Brain using SSE.",
            "GET /session/{session_id}": "See what Thinnu ate.",
            "GET /session/{session_id}/chunks": "See all chunks generated.",
            "GET /session/{session_id}/history": "See conversation history.",
            "DELETE /session/{session_id}/history": "Clear conversation history.",
            "DELETE /session/{session_id}": "Make Thinnu forget.",
            "GET /health": "Check if Ollama is reachable.",
        },
    }


@app.get("/health")
async def health():
    """
    Always check this first if something seems wrong.
    """
    ollama_ok = await check_ollama_connection()
    return {
        "status": "ok" if ollama_ok else "degraded",
        "ollama": "reachable" if ollama_ok else "unreachable — run: ollama serve",
        "sessions_active": len(sessions),
    }


@app.post("/scrape")
async def scrape(req: ScrapeRequest):
    """
    Full RAG pipeline on scrape:
    1. Thinnu eats the URL
    2. Chunker splits into ~500 word pieces
    3. Embedder converts every chunk into a vector
    4. Everything stored in session
    """
    # Step 1 — Thinnu eats
    try:
        data = await thinnu_eat(req.url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Thinnu choked: {str(e)}")

    # Step 2 — Chunk
    try:
        chunks_raw = chunk_scraped_data(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chunker failed: {str(e)}")

    # Step 3 — Embed
    try:
        embedded_chunks = await embed_chunks(chunks_raw)
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Brain unavailable: Cannot connect to Ollama at {str(e)}. Start it with: ollama serve",
        )

    # Step 4 — Store session
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "data": data,
        "embedded_chunks": embedded_chunks,
        "history": [],  # conversation history starts empty
    }

    return {
        "session_id": session_id,
        "url": data["url"],
        "title": data["title"],
        "message": "Thinnu ate it. Chunks embedded. Now ask anything.",
        "stats": {
            "headings": len(data["headings"]),
            "links": len(data["links"]),
            "images": len(data["images"]),
            "tables": len(data["tables"]),
            "forms": len(data["forms"]),
            "visible_text_chars": len(data["visible_text"]),
            "raw_html_chars": len(data["raw_html"]),  # stored but not embedded
            "total_chunks": len(embedded_chunks),
        },
    }


@app.post("/ask")
async def ask(req: AskRequest):
    """
    RAG retrieval + generation with conversation history.
    """
    if req.session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Feed Thinnu a URL first via /scrape.",
        )

    embedded_chunks = sessions[req.session_id]["embedded_chunks"]
    history = sessions[req.session_id]["history"]

    # Find relevant chunks
    try:
        relevant_chunks = await find_relevant_chunks(
            question=req.question,
            embedded_chunks=embedded_chunks,
            top_k=8,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Embedder failed: {str(e)}")

    # Answer with history context
    try:
        answer = await brain_think_with_history(relevant_chunks, req.question, history)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Brain failed: {str(e)}")

    # Save to history
    sessions[req.session_id]["history"].append({"role": "user", "content": req.question})
    sessions[req.session_id]["history"].append({"role": "assistant", "content": answer})

    return {
        "session_id": req.session_id,
        "question": req.question,
        "answer": answer,
        "chunks_used": len(relevant_chunks),
        "history_turns": len(sessions[req.session_id]["history"]) // 2,
    }


@app.post("/ask/stream")
async def ask_stream(req: AskRequest):
    """
    Streaming version of /ask with conversation history.
    """
    if req.session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Feed Thinnu a URL first via /scrape.",
        )

    embedded_chunks = sessions[req.session_id]["embedded_chunks"]
    history = sessions[req.session_id]["history"]

    # Find relevant chunks
    try:
        relevant_chunks = await find_relevant_chunks(
            question=req.question,
            embedded_chunks=embedded_chunks,
            top_k=8,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Embedder failed: {str(e)}")

    # Collect full answer for history
    full_answer = []

    async def token_generator():
        import json

        try:
            async for token in brain_think_stream_with_history(
                relevant_chunks, req.question, history
            ):
                full_answer.append(token)
                yield f"data: {json.dumps({'token': token})}\n\n"

            # Save to history after stream completes
            sessions[req.session_id]["history"].append(
                {"role": "user", "content": req.question}
            )
            sessions[req.session_id]["history"].append(
                {"role": "assistant", "content": "".join(full_answer)}
            )

            yield f"data: {json.dumps({'done': True, 'history_turns': len(sessions[req.session_id]['history']) // 2})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        token_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.delete("/session/{session_id}/history")
def clear_history(session_id: str):
    """
    Clear conversation history without losing scraped data.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    sessions[session_id]["history"] = []
    return {"message": "History cleared. Scraped data preserved."}


@app.get("/session/{session_id}/history")
def get_history(session_id: str):
    """
    View full conversation history for a session.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {
        "session_id": session_id,
        "history": sessions[session_id]["history"],
        "total_turns": len(sessions[session_id]["history"]) // 2,
    }


@app.get("/session/{session_id}")
def get_session(session_id: str):
    """
    Returns raw scraped data for a session.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    data = sessions[session_id]["data"]
    return {k: v for k, v in data.items() if k != "raw_html"}


@app.get("/session/{session_id}/chunks")
def get_chunks(session_id: str):
    """
    See all chunks generated from the scraped page.
    Useful for debugging what Brain is working with.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    chunks = sessions[session_id]["embedded_chunks"]
    return {
        "total_chunks": len(chunks),
        "chunks": [
            {"index": i, "preview": c["text"][:200] + "..."}
            for i, c in enumerate(chunks)
        ],
    }


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    """
    Clear a session from memory.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    del sessions[session_id]
    return {"message": "Thinnu has forgotten. Feed it again anytime."}
