"""
Scraper for bibliasacra.nl - downloads illustration data with images and iconclass codes.

Usage:
    python scraper.py [--output-dir OUTPUT_DIR] [--delay DELAY] [--no-images]
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://bibliasacra.nl"
BROWSE_URL = f"{BASE_URL}/browse/illustrations/iconclass/"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def get_soup(url: str, session: requests.Session) -> BeautifulSoup:
    """Fetch a URL and return a BeautifulSoup object."""
    resp = session.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def get_iconclass_links(session: requests.Session) -> list[dict]:
    """
    Parse the main browse page to get all iconclass category links and their counts.
    Returns list of dicts with 'code', 'url', 'count'.
    """
    soup = get_soup(BROWSE_URL, session)
    browse_list = soup.find("ul", id="browse-list")
    if not browse_list:
        # Fallback: find all links that point to iconclass subpages
        links = soup.find_all("a", href=re.compile(r"/browse/illustrations/iconclass/.+"))
    else:
        links = browse_list.find_all("a")

    results = []
    seen_urls = set()
    for link in links:
        href = link.get("href", "")
        if not href:
            continue
        url = urljoin(BASE_URL, href)
        if url in seen_urls:
            continue
        seen_urls.add(url)

        text = link.get_text(strip=True)
        # Extract count from brackets like [1112]
        count_match = re.search(r"\[(\d+)\]", text)
        count = int(count_match.group(1)) if count_match else 0
        # The code is the text before the bracket
        code = re.sub(r"\[\d+\]", "", text).strip().rstrip("*")

        results.append({"code": code, "url": url, "count": count})

    print(f"Found {len(results)} iconclass categories")
    return results


def parse_illustration_page(soup: BeautifulSoup, iconclass_code: str) -> list[dict]:
    """
    Parse a single iconclass subpage and extract all illustration data.
    Each illustration is in a <div class="document-part"> element.
    """
    illustrations = []

    # Try finding document-part divs
    parts = soup.find_all("div", class_="document-part")

    if parts:
        for part in parts:
            ill = parse_document_part(part, iconclass_code)
            if ill:
                illustrations.append(ill)
    else:
        # Fallback: parse table-like structures
        tables = soup.find_all("table")
        for table in tables:
            ill = parse_table_entry(table, iconclass_code)
            if ill:
                illustrations.append(ill)

        # If still nothing, try parsing from raw text/structure
        if not illustrations:
            illustrations = parse_from_structure(soup, iconclass_code)

    return illustrations


def parse_document_part(part, iconclass_code: str) -> dict | None:
    """Parse a single document-part div containing a listTable."""
    data = {"iconclass_code": iconclass_code}

    table = part.find("table", class_="listTable")
    if not table:
        return None

    rows = table.find_all("tr")
    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 2:
            continue

        label = cells[0].get_text(strip=True).lower()

        if "illustration id" in label:
            # The ID is in a listHead cell with a link
            head_cell = row.find("td", class_="listHead")
            if head_cell:
                link = head_cell.find("a")
                data["illustration_id"] = link.get_text(strip=True) if link else head_cell.get_text(strip=True)
                if link:
                    data["detail_url"] = urljoin(BASE_URL, link.get("href", ""))

            # The image is in the imghead cell (rowspan covers multiple rows)
            img_cell = row.find("td", class_="imghead")
            if img_cell:
                img = img_cell.find("img")
                if img:
                    src = img.get("src", "")
                    # Convert thumb URL to master URL for full resolution
                    thumb_url = urljoin(BASE_URL, src)
                    master_url = thumb_url.replace("/thumbs/", "/masters/")
                    data["image_url"] = master_url
                    data["image_thumb_url"] = thumb_url

                    # Extract image ID from URL
                    img_id_match = re.search(r"/(\d+)\.jpg", src)
                    if img_id_match:
                        data["image_id"] = img_id_match.group(1)

                # Also get reference text from img alt
                img = img_cell.find("img")
                if img and img.get("alt"):
                    data["reference"] = img.get("alt")

        elif "short title" in label:
            val_cell = row.find("td", class_="listValue")
            data["short_title"] = val_cell.get_text(strip=True) if val_cell else cells[1].get_text(strip=True)

        elif label == "type":
            val_cell = row.find("td", class_="listValue")
            data["type"] = val_cell.get_text(strip=True) if val_cell else cells[1].get_text(strip=True)

        elif "material" in label:
            val_cell = row.find("td", class_="listValue")
            data["material"] = val_cell.get_text(strip=True) if val_cell else cells[1].get_text(strip=True)

        elif "artist" in label:
            val_cell = row.find("td", class_="listValue")
            data["artist"] = val_cell.get_text(strip=True) if val_cell else cells[1].get_text(strip=True)

    if "illustration_id" in data:
        return data
    return None


def parse_table_entry(table, iconclass_code: str) -> dict | None:
    """Parse illustration data from a table element."""
    data = {"iconclass_code": iconclass_code}
    rows = table.find_all("tr")
    for row in rows:
        cells = row.find_all(["td", "th"])
        if len(cells) >= 2:
            label = cells[0].get_text(strip=True).lower()
            value = cells[1].get_text(strip=True)
            if "illustration id" in label:
                data["illustration_id"] = value
                if len(cells) >= 3:
                    data["reference"] = cells[2].get_text(strip=True)
            elif "short title" in label:
                data["short_title"] = value
            elif "type" in label:
                data["type"] = value
            elif "material" in label:
                data["material"] = value
            elif "artist" in label:
                data["artist"] = value

    img = table.find("img")
    if img:
        src = img.get("src", "")
        if src:
            data["image_url"] = urljoin(BASE_URL, src)

    if "illustration_id" in data:
        return data
    return None


def parse_from_structure(soup: BeautifulSoup, iconclass_code: str) -> list[dict]:
    """
    Fallback parser: tries to extract illustration data from any structure on the page.
    """
    illustrations = []
    # Find all text that looks like "Illustration ID"
    all_text = soup.get_text()
    # Split by "Illustration ID" occurrences
    parts = re.split(r"Illustration ID", all_text)

    for i, part in enumerate(parts[1:], 1):  # Skip first (before any ID)
        lines = [l.strip() for l in part.strip().split("\n") if l.strip()]
        if not lines:
            continue

        data = {"iconclass_code": iconclass_code}
        data["illustration_id"] = lines[0].strip() if lines else ""

        for j, line in enumerate(lines):
            lower = line.lower()
            if lower.startswith("short title"):
                data["short_title"] = line.replace("Short title", "").strip()
            elif lower.startswith("type"):
                data["type"] = line.replace("Type", "").strip()
            elif lower.startswith("material"):
                data["material"] = line.replace("Material", "").strip()
            elif lower.startswith("artist"):
                data["artist"] = line.replace("Artist", "").strip()

        if data.get("illustration_id"):
            illustrations.append(data)

    # Try to find images
    imgs = soup.find_all("img")
    for img in imgs:
        src = img.get("src", "")
        if "images" in src and src not in ["", None]:
            # Try to associate with nearest illustration
            # For now just collect them
            pass

    return illustrations


def check_pagination(soup: BeautifulSoup) -> list[str]:
    """Check if the page has pagination and return all page URLs."""
    pages = []
    pagination = soup.find("ul", class_="pagination") or soup.find("nav", class_="pagination")
    if pagination:
        for a in pagination.find_all("a"):
            href = a.get("href", "")
            if href:
                pages.append(urljoin(BASE_URL, href))
    return list(set(pages))


def download_image(url: str, output_dir: Path, session: requests.Session) -> str | None:
    """Download an image and return the local filename."""
    try:
        filename = url.split("/")[-1]
        filepath = output_dir / filename
        if filepath.exists():
            return filename

        resp = session.get(url, headers=HEADERS, timeout=30, stream=True)
        resp.raise_for_status()

        with open(filepath, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        return filename
    except Exception as e:
        print(f"  Failed to download image {url}: {e}")
        return None


def scrape_iconclass_page(
    url: str,
    iconclass_code: str,
    session: requests.Session,
    delay: float = 1.0,
) -> list[dict]:
    """Scrape all illustrations from a single iconclass page (handling pagination)."""
    all_illustrations = []
    visited = set()
    urls_to_visit = [url]

    while urls_to_visit:
        current_url = urls_to_visit.pop(0)
        if current_url in visited:
            continue
        visited.add(current_url)

        try:
            soup = get_soup(current_url, session)
        except Exception as e:
            print(f"  Error fetching {current_url}: {e}")
            continue

        illustrations = parse_illustration_page(soup, iconclass_code)
        all_illustrations.extend(illustrations)

        # Check for pagination
        page_urls = check_pagination(soup)
        for pu in page_urls:
            if pu not in visited:
                urls_to_visit.append(pu)

        time.sleep(delay)

    return all_illustrations


def main():
    parser = argparse.ArgumentParser(description="Scrape bibliasacra.nl illustrations")
    parser.add_argument("--output-dir", default="scraped_data", help="Output directory")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests (seconds)")
    parser.add_argument("--no-images", action="store_true", help="Skip downloading images")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of categories to scrape (0 = all)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing data.json")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    if not args.no_images:
        images_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()

    # Load existing data if resuming
    data_file = output_dir / "data.json"
    all_data = []
    scraped_codes = set()
    if args.resume and data_file.exists():
        with open(data_file) as f:
            all_data = json.load(f)
        scraped_codes = {d["iconclass_code"] for d in all_data}
        print(f"Resuming: {len(all_data)} illustrations already scraped from {len(scraped_codes)} categories")

    # Step 1: Get all iconclass category links
    print("Fetching iconclass category links...")
    categories = get_iconclass_links(session)
    time.sleep(args.delay)

    if args.limit > 0:
        categories = categories[:args.limit]

    # Step 2: Scrape each category
    total = len(categories)
    for i, cat in enumerate(categories, 1):
        code = cat["code"]
        if code in scraped_codes:
            print(f"[{i}/{total}] Skipping {code} (already scraped)")
            continue

        print(f"[{i}/{total}] Scraping iconclass '{code}' ({cat['count']} illustrations)...")
        illustrations = scrape_iconclass_page(cat["url"], code, session, args.delay)

        # Download images
        if not args.no_images:
            for ill in illustrations:
                if "image_url" in ill:
                    filename = download_image(ill["image_url"], images_dir, session)
                    if filename:
                        ill["image_file"] = filename
                    time.sleep(0.5)

        all_data.extend(illustrations)
        print(f"  Found {len(illustrations)} illustrations")

        # Save incrementally
        with open(data_file, "w") as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)

    # Final save
    with open(data_file, "w") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

    print(f"\nDone! Scraped {len(all_data)} illustrations total.")
    print(f"Data saved to {data_file}")
    if not args.no_images:
        print(f"Images saved to {images_dir}")


if __name__ == "__main__":
    main()
