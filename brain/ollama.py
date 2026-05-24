import json
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


async def brain_think_stream(relevant_chunks: list[str], question: str):
    """
    Streaming version of brain_think.
    Yields tokens one by one as Mistral generates them.
    """
    context = "\n\n---\n\n".join(relevant_chunks)

    prompt = f"""You are an intelligent assistant analyzing scraped webpage content.
You have been given the most relevant sections of a webpage that relate to the user's question.
Answer accurately and thoroughly using ONLY the content provided below.
If the answer is not in the content, say \"I could not find that information on this page.\"
Do NOT use your training data. Do NOT make things up.

=== RELEVANT PAGE CONTENT ===
{context}
=== END OF CONTENT ===

User's question: {question}

Answer:"""

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": True
            }
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.strip():
                    try:
                        data = json.loads(line)
                        token = data.get("response", "")
                        done = data.get("done", False)
                        if token:
                            yield token
                        if done:
                            break
                    except json.JSONDecodeError:
                        continue


async def brain_think_with_history(
    relevant_chunks: list[str],
    question: str,
    history: list[dict]
) -> str:
    """
    Answers with full conversation history context.
    History format: [{"role": "user/assistant", "content": "..."}]
    """
    context = "\n\n---\n\n".join(relevant_chunks)

    # Build history string
    history_str = ""
    if history:
        history_str = "\n=== CONVERSATION HISTORY ===\n"
        for turn in history[-6:]:  # last 6 turns max (3 exchanges)
            role = "User" if turn["role"] == "user" else "Assistant"
            history_str += f"{role}: {turn['content']}\n"
        history_str += "=== END OF HISTORY ===\n"

    prompt = f"""You are an intelligent assistant analyzing scraped webpage content.
Answer using ONLY the page content provided. Use conversation history for context on follow-up questions.
If the answer is not in the content, say "I could not find that information on this page."
Do NOT use your training data. Do NOT make things up.

=== RELEVANT PAGE CONTENT ===
{context}
=== END OF CONTENT ===
{history_str}
Current question: {question}

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


async def brain_think_stream_with_history(
    relevant_chunks: list[str],
    question: str,
    history: list[dict]
):
    """
    Streaming version with conversation history.
    """
    context = "\n\n---\n\n".join(relevant_chunks)

    history_str = ""
    if history:
        history_str = "\n=== CONVERSATION HISTORY ===\n"
        for turn in history[-6:]:
            role = "User" if turn["role"] == "user" else "Assistant"
            history_str += f"{role}: {turn['content']}\n"
        history_str += "=== END OF HISTORY ===\n"

    prompt = f"""You are an intelligent assistant analyzing scraped webpage content.
Answer using ONLY the page content provided. Use conversation history for context on follow-up questions.
If the answer is not in the content, say "I could not find that information on this page."
Do NOT use your training data. Do NOT make things up.

=== RELEVANT PAGE CONTENT ===
{context}
=== END OF CONTENT ===
{history_str}
Current question: {question}

Answer:"""

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": True
            }
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.strip():
                    try:
                        data = json.loads(line)
                        token = data.get("response", "")
                        done = data.get("done", False)
                        if token:
                            yield token
                        if done:
                            break
                    except json.JSONDecodeError:
                        continue


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