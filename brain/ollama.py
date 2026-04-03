import httpx
import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")


async def brain_think(relevant_chunks: list[str], question: str) -> str:
    """
    Mistral only sees the top_k relevant chunks — not the entire page.
    This is what makes it stop hallucinating.
    """
    context = "\n\n---\n\n".join(relevant_chunks)

    prompt = f"""You are an intelligent assistant analyzing scraped webpage content.
You have been given the most relevant sections of a webpage that relate to the user's question.
Answer accurately and thoroughly using ONLY the content provided below.
If the answer is not in the content, say "I could not find that information on this page."
Do NOT use your training data. Do NOT make things up.

=== RELEVANT PAGE CONTENT ===
{context}
=== END OF CONTENT ===

User's question: {question}

Answer:"""

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            }
        )
        response.raise_for_status()
        return response.json()["response"]


async def check_ollama_connection() -> bool:
    """
    Health check — is Ollama reachable?
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{OLLAMA_URL}")
            return response.status_code == 200
    except Exception:
        return False