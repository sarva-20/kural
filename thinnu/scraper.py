import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup


async def thinnu_eat(url: str) -> dict:
    """
    Thinnu sees a URL.
    Thinnu eats the URL.
    Thinnu leaves nothing behind.
    """

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        # Go to the page, wait for all network calls to settle
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)

# Scroll down to trigger lazy-loaded content
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await asyncio.sleep(1)
        await page.evaluate("window.scrollTo(0, 0)")
        await asyncio.sleep(2)  # let JS finish its drama

        html = await page.content()
        title = await page.title()
        final_url = page.url

        await browser.close()

    soup = BeautifulSoup(html, "lxml")

    # --- Strip noise ---
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # --- Visible text ---
    visible_text = soup.get_text(separator="\n", strip=True)

    # --- Meta tags ---
    meta = {}
    for tag in soup.find_all("meta"):
        name = tag.get("name") or tag.get("property") or tag.get("http-equiv")
        content = tag.get("content")
        if name and content:
            meta[name] = content

    # --- Headings ---
    headings = []
    for level in ["h1", "h2", "h3", "h4", "h5", "h6"]:
        for h in soup.find_all(level):
            headings.append({"level": level, "text": h.get_text(strip=True)})

    # --- Links ---
    links = []
    for a in soup.find_all("a", href=True):
        links.append({"text": a.get_text(strip=True), "href": a["href"]})

    # --- Images ---
    images = []
    for img in soup.find_all("img"):
        images.append({
            "src": img.get("src", ""),
            "alt": img.get("alt", ""),
            "title": img.get("title", "")
        })

    # --- Tables ---
    tables = []
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if cells:
                rows.append(cells)
        if rows:
            tables.append(rows)

    # --- Forms ---
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

    # --- Raw HTML (capped) ---
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