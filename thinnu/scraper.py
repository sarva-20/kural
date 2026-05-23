import asyncio
import sys
import threading
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup


def thinnu_eat_sync(url: str) -> dict:
    """
    Synchronous version using sync_playwright.
    Runs in a separate thread to avoid event loop conflicts on Windows.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        page.goto(url, wait_until="domcontentloaded", timeout=30000)

        # Scroll to trigger lazy loading
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        import time
        time.sleep(1)
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(2)

        html = page.content()
        title = page.title()
        final_url = page.url

        browser.close()

    soup = BeautifulSoup(html, "lxml")

    # Strip noise
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    visible_text = soup.get_text(separator="\n", strip=True)

    # Meta tags
    meta = {}
    for tag in soup.find_all("meta"):
        name = tag.get("name") or tag.get("property") or tag.get("http-equiv")
        content = tag.get("content")
        if name and content:
            meta[name] = content

    # Headings
    headings = []
    for level in ["h1", "h2", "h3", "h4", "h5", "h6"]:
        for h in soup.find_all(level):
            headings.append({"level": level, "text": h.get_text(strip=True)})

    # Links
    links = []
    for a in soup.find_all("a", href=True):
        links.append({"text": a.get_text(strip=True), "href": a["href"]})

    # Images
    images = []
    for img in soup.find_all("img"):
        images.append({
            "src": img.get("src", ""),
            "alt": img.get("alt", ""),
            "title": img.get("title", "")
        })

    # Tables
    tables = []
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if cells:
                rows.append(cells)
        if rows:
            tables.append(rows)

    # Forms
    forms = []
    for form in soup.find_all("form"):
        fields = []
        for inp in form.find_all(["input", "textarea", "select", "button"]):
            fields.append({
                "tag": inp.name,
                "type": inp.get("type", ""),
                "name": inp.get("name", ""),
                "placeholder": inp.get("placeholder", ""),
                "value": inp.get("value", "")
            })
        forms.append({
            "action": form.get("action", ""),
            "method": form.get("method", "get"),
            "fields": fields
        })

    raw_html = html[:50000]

    return {
        "url": final_url,
        "title": title,
        "meta": meta,
        "headings": headings,
        "visible_text": visible_text[:30000],
        "links": links,
        "images": images,
        "tables": tables,
        "forms": forms,
        "raw_html": raw_html
    }


async def thinnu_eat(url: str) -> dict:
    """
    Async wrapper that runs sync Playwright in a thread pool.
    This is the correct way to use Playwright on Windows with asyncio.
    """
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, thinnu_eat_sync, url)
    return result