def chunk_text(text: str, chunk_size: int = 200, overlap: int = 20) -> list[str]:
    """
    Splits text into overlapping chunks.

    Why overlap? So context isn't lost at chunk boundaries.
    Example: A sentence split across two chunks is still readable in both.

    chunk_size: ~500 words per chunk (safe for embedding models)
    overlap: last 50 words of chunk N become first 50 words of chunk N+1
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # slide forward with overlap

    return chunks


def chunk_scraped_data(data: dict) -> list[str]:
    """
    Takes everything Thinnu ate and builds a clean
    list of chunks ready for embedding.

    We chunk different sections separately so context
    doesn't bleed between headings, links, and body text.
    """
    all_chunks = []

    identity = f"Page Title: {data['title']}\nURL: {data['url']}"
    if data["meta"]:
        meta_str = "\n".join([f"{k}: {v}" for k, v in data["meta"].items()])
        identity += f"\nMeta:\n{meta_str}"
    all_chunks.append(identity[:800])  # cap at 800 chars

    # --- Headings (one chunk, gives structural overview) ---
    if data["headings"]:
        headings_str = "\n".join(
            [f"[{h['level'].upper()}] {h['text']}" for h in data["headings"]]
        )
        all_chunks.append(f"Page Structure (Headings):\n{headings_str}")

    # --- Visible text (chunked, this is the meat) ---
    if data["visible_text"]:
        text_chunks = chunk_text(data["visible_text"], chunk_size=200, overlap=20)
        all_chunks.extend(text_chunks)

    # --- Links (one chunk, capped at 50 and 1000 chars) ---
    if data["links"]:
        links_str = "\n".join(
            [f"- {l['text']} -> {l['href']}" for l in data["links"][:50]]
        )
        links_str = links_str[:1000]  # hard cap at 1000 chars
        all_chunks.append(f"Links on page:\n{links_str}")

    # --- Images (one chunk) ---
    if data["images"]:
        images_str = "\n".join(
            [f"- src: {img['src']} alt: {img['alt']}" for img in data["images"][:50]]
        )
        all_chunks.append(f"Images on page:\n{images_str}")

    # --- Tables (each table is its own chunk) ---
    for i, table in enumerate(data["tables"]):
        rows_str = "\n".join([" | ".join(row) for row in table])
        all_chunks.append(f"Table {i+1}:\n{rows_str}")

    # --- Forms (one chunk) ---
    if data["forms"]:
        forms_str = ""
        for form in data["forms"]:
            forms_str += f"Form action={form['action']} method={form['method']}\n"
            for field in form["fields"]:
                forms_str += f"  - {field['tag']} name={field['name']} type={field['type']} placeholder={field['placeholder']}\n"
        all_chunks.append(f"Forms on page:\n{forms_str}")

   

    # Filter out empty chunks
    all_chunks = [c.strip() for c in all_chunks if c.strip()]

    return all_chunks