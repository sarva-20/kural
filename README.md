# Kural 🔍

> *குறள் — distilled wisdom*

A production-grade web scraper with a local RAG brain. Give it any URL, ask anything about that page. No cloud. No API bills. Fully offline intelligence.

---

## What it does
URL → Thinnu eats the page → chunks → embeddings → you ask → relevant chunks retrieved → Mistral answers

No hallucination. No context overflow. Just the page content, retrieved semantically, answered locally.

---

## Architecture

```text
┌─────────────────────────────────────────────────┐
│                    Kural v2.0                    │
├─────────────────────────────────────────────────┤
│                                                  │
│  POST /scrape                                    │
│    └── Thinnu (Playwright + BS4)                 │
│          └── Chunker (200-word overlapping)      │
│                └── nomic-embed-text (Ollama)     │
│                      └── Session store           │
│                                                  │
│  POST /ask                                       │
│    └── Embed question (nomic-embed-text)         │
│          └── Cosine similarity → top 5 chunks    │
│                └── Mistral 7B (Ollama)           │
│                      └── Grounded answer         │
│                                                  │
└─────────────────────────────────────────────────┘
```

---

## Stack


| Layer | Technology |
|:---|:---|
| **API** | FastAPI + Uvicorn |
| **Scraping** | Playwright (headless Chromium) + BeautifulSoup4 |
| **Embeddings** | nomic-embed-text via Ollama |
| **Retrieval** | Cosine similarity (numpy) |
| **LLM** | Mistral 7B via Ollama |
| **Runtime** | Python 3.11+ |

---

## Project Structure

```text
kural/
├── thinnu/
│   ├── __init__.py
│   └── scraper.py        # Playwright + BS4 scraper
├── brain/
│   ├── __init__.py
│   ├── chunker.py        # 200-word overlapping chunker
│   ├── embedder.py       # nomic-embed-text + cosine similarity
│   └── ollama.py         # Mistral 7B interface
├── main.py               # FastAPI app + all endpoints
├── requirements.txt
└── .env                  # not committed
```

---

## What Thinnu extracts

Every single thing on the page:

*   Visible text (stripped of scripts/styles)
*   H1–H6 headings with hierarchy
*   All links (text + href)
*   Images (src + alt + title)
*   Tables (full row/cell extraction)
*   Forms (all inputs, types, names, placeholders)
*   Meta tags (name, property, content)
*   Raw HTML (first 50k chars, stored not embedded)

---

## Setup

### Prerequisites

*   Python 3.11+
*   [Ollama](https://ollama.ai) running locally
*   Mistral 7B and nomic-embed-text pulled

```bash
ollama pull mistral
ollama pull nomic-embed-text
ollama serve
```

### Install

```bash
git clone https://github.com/sarva-20/kural.git
cd kural
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium
```

### Configure

Create a `.env` file in the root directory:

```env
OLLAMA_URL=http://127.0.0.1:11434
OLLAMA_MODEL=mistral
EMBED_MODEL=nomic-embed-text
```

### Run

```bash
uvicorn main:app --reload --port 8000
```

---

## API

### `GET /health`
Check if Ollama is reachable.

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "ollama": "reachable",
  "sessions_active": 0
}
```

---

### `POST /scrape`
Feed Thinnu a URL. Returns a `session_id`.

```bash
curl -X POST http://localhost:8000/scrape \
  -H "Content-Type: application/json" \
  -d '{"url": "https://en.wikipedia.org/wiki/Mistral_AI"}'
```

```json
{
  "session_id": "abc-123",
  "title": "Mistral AI - Wikipedia",
  "message": "Thinnu ate it. Chunks embedded. Now ask anything.",
  "stats": {
    "total_chunks": 38,
    "visible_text_chars": 28961
  }
}
```

---

### `POST /ask`
Ask anything. RAG retrieves, Mistral answers.

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "abc-123",
    "question": "When was Mistral AI founded and who founded it?"
  }'
```

```json
{
  "answer": "Mistral AI was founded in April 2023 by Arthur Mensch, Guillaume Lample, and Timothée Lacroix.",
  "chunks_used": 5
}
```

---

### `GET /session/{session_id}/chunks`
Debug — see all chunks generated from the scraped page.

### `DELETE /session/{session_id}`
Clear a session from memory.

---

## RAG Pipeline — Why it works

Naive approach: stuff all 80k chars of scraped content into the LLM context → context overflow → hallucination.

Kural's approach:
1. Split content into 200-word chunks with 20-word overlap.
2. Embed every chunk with `nomic-embed-text` (274MB, runs locally).
3. On question → embed the question → cosine similarity against all chunks.
4. Send only top 5 relevant chunks to Mistral 7B.
5. Mistral answers from actual page content, not training data.

Result: Mistral 7B with 32k context window handles any page correctly because it never sees more than ~1500 words at a time.

---

## Tested on

*   Wikipedia articles :white_check_mark:
*   Documentation pages :white_check_mark:
*   Static blogs and news articles :white_check_mark:

**Known limitations:**
*   JS-heavy SPAs (React, Next.js) — partial content only.
*   GitHub profile pages — repo list is dynamically loaded.
*   Sites with aggressive anti-bot protection (TechCrunch, etc.).

---

## What's next

*   **Module 2** — Streaming responses via SSE
*   **Module 3** — Chat history per session
*   **Module 4** — Multi-page BFS crawling
*   **Module 5** — Resilience layer (retries, anti-bot evasion)
*   **Module 6** — Redis persistent sessions
*   **Module 7** — API key authentication
*   **Module 8** — Structured logging with loguru
*   **Module 9** — Docker + docker-compose
*   **Kural fine-tune** — QLoRA fine-tuned Llama 3.2 3B for structured JSON extraction from scraped chunks

---

## Author

**Sarvatarshan Sankar** — [@sarva-20](https://github.com/sarva-20)

Built as part of a production-grade learning project. Named after Thirukkural — distilled wisdom.
