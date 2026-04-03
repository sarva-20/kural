import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from thinnu.scraper import thinnu_eat
from brain.chunker import chunk_scraped_data
from brain.embedder import embed_chunks, find_relevant_chunks
from brain.ollama import brain_think, check_ollama_connection

app = FastAPI(
    title="Theduvaan",
    description="Thinnu eats the web. Brain understands it. You ask anything.",
    version="2.0.0"
)

# In-memory session store
# session_id -> { "data": raw scraped data, "chunks": embedded chunks }
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
        "name": "Theduvaan 🔍",
        "tagline": "Thinnu eats. Brain thinks. You ask.",
        "version": "2.0.0",
        "architecture": "RAG (Retrieval Augmented Generation)",
        "models": {
            "chat": "mistral:7b",
            "embeddings": "nomic-embed-text"
        },
        "endpoints": {
            "POST /scrape": "Give Thinnu a URL. It eats, chunks, and embeds everything.",
            "POST /ask": "Ask the Brain anything. RAG retrieves, Mistral answers.",
            "GET /session/{session_id}": "See what Thinnu ate.",
            "GET /session/{session_id}/chunks": "See all chunks generated.",
            "DELETE /session/{session_id}": "Make Thinnu forget.",
            "GET /health": "Check if Ollama is reachable."
        }
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
        "sessions_active": len(sessions)
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
            detail=f"Brain unavailable: Cannot connect to Ollama at {str(e)}. Start it with: ollama serve"
        )

    # Step 4 — Store session
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "data": data,
        "embedded_chunks": embedded_chunks
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
            "raw_html_chars": len(data["raw_html"]), # stored but not embedded
            "total_chunks": len(embedded_chunks)
        }
    }


@app.post("/ask")
async def ask(req: AskRequest):
    """
    RAG retrieval + generation:
    1. Embed the question
    2. Find top 5 most relevant chunks via cosine similarity
    3. Send only those chunks to Mistral
    4. Mistral answers from actual page content
    """
    if req.session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Feed Thinnu a URL first via /scrape."
        )

    embedded_chunks = sessions[req.session_id]["embedded_chunks"]

    # Step 1+2 — Find relevant chunks
    try:
        relevant_chunks = await find_relevant_chunks(
            question=req.question,
            embedded_chunks=embedded_chunks,
            top_k=5
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Embedder failed: {str(e)}")

    # Step 3 — Mistral answers
    try:
        answer = await brain_think(relevant_chunks, req.question)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Brain failed: {str(e)}")

    return {
        "session_id": req.session_id,
        "question": req.question,
        "answer": answer,
        "chunks_used": len(relevant_chunks)
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
        ]
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