import httpx
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")


async def embed_text(text: str) -> list[float]:
    """
    Converts a piece of text into a vector (list of floats).
    This is the magic — similar texts get similar vectors.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={
                "model": EMBED_MODEL,
                "prompt": text
            }
        )
        response.raise_for_status()
        return response.json()["embedding"]


async def embed_chunks(chunks: list[str]) -> list[dict]:
    embedded = []
    for i, chunk in enumerate(chunks):
        try:
            embedding = await embed_text(chunk)
            embedded.append({
                "text": chunk,
                "embedding": embedding
            })
        except Exception as e:
            print(f"  ✗ Chunk {i} failed ({len(chunk)} chars): {str(e)[:80]}")
            print(f"    Preview: {chunk[:100]}")
            # Skip failed chunks instead of crashing
            continue
    print(f"  ✓ Embedded {len(embedded)} chunks successfully")
    return embedded


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Measures how similar two vectors are.
    1.0 = identical meaning
    0.0 = completely unrelated
    -1.0 = opposite meaning

    This is how we find which chunks are relevant to the question.
    No database needed — pure math.
    """
    a = np.array(vec_a)
    b = np.array(vec_b)
    
    dot_product = np.dot(a, b)
    magnitude = np.linalg.norm(a) * np.linalg.norm(b)
    
    if magnitude == 0:
        return 0.0
    
    return float(dot_product / magnitude)


async def find_relevant_chunks(
    question: str,
    embedded_chunks: list[dict],
    top_k: int = 5
) -> list[str]:
    """
    The core RAG retrieval step.

    1. Embed the question into a vector
    2. Compare it against every chunk's vector using cosine similarity
    3. Return the top_k most similar chunks

    Mistral only sees these top_k chunks — not the entire page.
    That's why it stops hallucinating.
    """
    question_embedding = await embed_text(question)

    # Score every chunk
    scored = []
    for chunk in embedded_chunks:
        score = cosine_similarity(question_embedding, chunk["embedding"])
        scored.append({
            "text": chunk["text"],
            "score": score
        })

    # Sort by similarity score, highest first
    scored.sort(key=lambda x: x["score"], reverse=True)

    # Return just the text of top_k chunks
    top_chunks = [item["text"] for item in scored[:top_k]]

    return top_chunks