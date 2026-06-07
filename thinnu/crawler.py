import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin, urlparse

from thinnu.scraper import thinnu_eat_sync


def get_internal_links(data: dict, base_url: str) -> list[str]:
    """
    Extracts internal links from scraped data.
    Only keeps links that stay within the same domain.
    """
    base_domain = urlparse(base_url).netloc
    internal_links = []

    for link in data.get("links", []):
        href = link.get("href", "")

        # Skip empty, anchors, mailto, javascript
        if not href or href.startswith(("#", "mailto:", "javascript:", "tel:")):
            continue

        # Make absolute URL
        absolute = urljoin(base_url, href)
        parsed = urlparse(absolute)

        # Only keep same domain, http/https
        if parsed.netloc == base_domain and parsed.scheme in ("http", "https"):
            # Strip fragments
            clean = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            if parsed.query:
                clean += f"?{parsed.query}"
            internal_links.append(clean)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for link in internal_links:
        if link not in seen:
            seen.add(link)
            unique.append(link)

    return unique


async def crawl_site(
    start_url: str,
    max_pages: int = 10,
    max_depth: int = 2
) -> list[dict]:
    """
    BFS crawler. Starts at start_url, discovers internal links,
    scrapes each page up to max_pages and max_depth.

    Returns list of scraped data dicts — one per page.
    """
    visited = set()
    queue = deque([(start_url, 0)])  # (url, depth)
    results = []

    print(f"\n[Crawler] Starting BFS from {start_url}")
    print(f"[Crawler] Max pages: {max_pages} | Max depth: {max_depth}")

    loop = asyncio.get_event_loop()

    with ThreadPoolExecutor(max_workers=1) as executor:
        while queue and len(results) < max_pages:
            url, depth = queue.popleft()

            # Skip if already visited
            if url in visited:
                continue

            visited.add(url)
            print(f"[Crawler] [{len(results)+1}/{max_pages}] depth={depth} {url}")

            # Scrape the page
            try:
                data = await loop.run_in_executor(
                    executor,
                    thinnu_eat_sync,
                    url
                )
                results.append(data)
                print(f"[Crawler] ✓ {data['title'][:50]} | {len(data['visible_text'])} chars")
            except Exception as e:
                print(f"[Crawler] ✗ Failed: {e}")
                continue

            # Discover links if not at max depth
            if depth < max_depth:
                links = get_internal_links(data, start_url)
                for link in links:
                    if link not in visited:
                        queue.append((link, depth + 1))

            # Small delay between requests — be polite
            await asyncio.sleep(1)

    print(f"[Crawler] Done. Scraped {len(results)} pages.")
    return results
