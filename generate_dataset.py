import asyncio
import json
import httpx
import re
import sys
import os
from datetime import datetime
from pathlib import Path

# Add project root to path so we can import thinnu and brain
sys.path.append(str(Path(__file__).parent))

from thinnu.scraper import thinnu_eat
from brain.chunker import chunk_scraped_data

# ── Config ────────────────────────────────────────────────────────────────────

OLLAMA_URL = "http://127.0.0.1:11434"
LABELER_MODEL = "mistral"
OUTPUT_FILE = "dataset.jsonl"
MIN_CHUNK_LENGTH = 50   # skip chunks shorter than this
MAX_CHUNK_LENGTH = 1500 # skip chunks longer than this

# ── Seed URLs ─────────────────────────────────────────────────────────────────
# Diverse domains = diverse training data = better generalization

SEED_URLS = [
    "https://docs.python.org/3/tutorial/classes.html",
    "https://docs.python.org/3/tutorial/errors.html",
    "https://docs.python.org/3/tutorial/modules.html",
    "https://fastapi.tiangolo.com/tutorial/dependencies/",
    "https://fastapi.tiangolo.com/tutorial/security/first-steps/",
    "https://numpy.org/doc/stable/user/basics.indexing.html",
    "https://scikit-learn.org/stable/modules/cross_validation.html",
    "https://huggingface.co/docs/transformers/preprocessing",
    "https://huggingface.co/docs/peft/tutorial/peft_model_config",
    "https://docs.docker.com/get-started/04_sharing_app/",
]
# ── Labeler prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise information extraction system.
Given a raw web content chunk, extract structured information as JSON.
Output ONLY valid JSON. No explanation. No markdown. No backticks. Just the JSON object."""

def build_labeling_prompt(chunk: str) -> str:
    return f"""Extract structured information from this web content chunk.

Return a JSON object with exactly these fields:
- "title": the main topic or subject of this chunk (string)
- "summary": 1-2 sentence summary of the chunk content (string)  
- "links": list of any URLs mentioned in the chunk (list of strings, empty list if none)
- "entities": list of named entities — people, companies, places, products, technologies (list of strings)

Web content chunk:
{chunk}

JSON output:"""

# ── Ollama labeler ────────────────────────────────────────────────────────────

async def label_chunk(chunk: str, client: httpx.AsyncClient) -> dict | None:
    """
    Send a chunk to Mistral 7B for JSON labeling.
    Returns parsed JSON dict or None if invalid.
    """
    prompt = build_labeling_prompt(chunk)

    try:
        response = await client.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": LABELER_MODEL,
                "prompt": f"<s>[INST] {SYSTEM_PROMPT}\n\n{prompt} [/INST]",
                "stream": False,
                "options": {
                    "temperature": 0.1,  # low temp = consistent JSON output
                    "top_p": 0.9,
                }
            },
            timeout=60.0
        )
        response.raise_for_status()
        raw = response.json()["response"].strip()

        # Strip markdown code fences if model adds them
        raw = re.sub(r"```json\s*", "", raw)
        raw = re.sub(r"```\s*", "", raw)
        raw = raw.strip()

        parsed = json.loads(raw)

        # Validate required fields exist
        required = {"title", "summary", "links", "entities"}
        if not required.issubset(parsed.keys()):
            return None

        # Validate types
        if not isinstance(parsed["title"], str):
            return None
        if not isinstance(parsed["summary"], str):
            return None
        if not isinstance(parsed["links"], list):
            return None
        if not isinstance(parsed["entities"], list):
            return None

        return parsed

    except (json.JSONDecodeError, KeyError, httpx.HTTPError):
        return None

# ── Alpaca formatter ──────────────────────────────────────────────────────────

def to_alpaca(chunk: str, label: dict) -> dict:
    """
    Converts a chunk + label into Alpaca training format.
    This is the format Unsloth's SFTTrainer expects.
    """
    return {
        "instruction": "Extract structured information from this web content chunk as JSON with fields: title, summary, links, entities.",
        "input": chunk,
        "output": json.dumps(label, ensure_ascii=False)
    }

# ── Main pipeline ─────────────────────────────────────────────────────────────

async def generate_dataset():
    print(f"\n{'='*60}")
    print(f"  Kural Dataset Generator")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  URLs: {len(SEED_URLS)}")
    print(f"  Output: {OUTPUT_FILE}")
    print(f"{'='*60}\n")

    total_chunks = 0
    total_labeled = 0
    total_failed = 0

    # Load existing samples first — NEVER overwrite
    samples = []
    if Path(OUTPUT_FILE).exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        print(f"✓ Loaded {len(samples)} existing samples from {OUTPUT_FILE}\n")

    async with httpx.AsyncClient() as client:
        for i, url in enumerate(SEED_URLS, 1):
            print(f"[{i}/{len(SEED_URLS)}] Scraping: {url}")

            # Step 1 — Thinnu eats
            try:
                data = await thinnu_eat(url)
            except Exception as e:
                print(f"  ✗ Scrape failed: {e}")
                continue

            # Step 2 — Chunk
            chunks = chunk_scraped_data(data)
            # Filter chunks by length
            chunks = [
                c for c in chunks
                if MIN_CHUNK_LENGTH <= len(c) <= MAX_CHUNK_LENGTH
            ]

            print(f"  ✓ Scraped: {data['title'][:50]} | {len(chunks)} chunks")
            total_chunks += len(chunks)

            # Step 3 — Label each chunk
            url_labeled = 0
            for j, chunk in enumerate(chunks):
                label = await label_chunk(chunk, client)

                if label:
                    sample = to_alpaca(chunk, label)
                    samples.append(sample)
                    url_labeled += 1
                    total_labeled += 1
                else:
                    total_failed += 1

                # Progress indicator every 10 chunks
                if (j + 1) % 10 == 0:
                    print(f"    → {j+1}/{len(chunks)} chunks labeled...")

            print(f"  ✓ Labeled: {url_labeled}/{len(chunks)} chunks")

            # Save incrementally — don't lose progress if it crashes
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                for sample in samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            print(f"  ✓ Saved {len(samples)} total samples\n")

    # Final summary
    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"  Total chunks processed : {total_chunks}")
    print(f"  Successfully labeled   : {total_labeled}")
    print(f"  Failed / skipped       : {total_failed}")
    print(f"  Success rate           : {total_labeled/max(total_chunks,1)*100:.1f}%")
    print(f"  Dataset saved to       : {OUTPUT_FILE}")
    print(f"{'='*60}\n")

    return samples


if __name__ == "__main__":
    asyncio.run(generate_dataset())