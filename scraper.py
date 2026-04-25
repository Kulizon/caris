"""
Scraper for bibliasacra.nl - downloads illustration data with images and iconclass codes.

Two modes:
  Mode 1 (default): Scrape category listing pages to build the base dataset.
      python scraper.py --output-dir scraped_data [--limit N]

  Mode 2 (detail enrichment): Fetch each illustration's detail page + download images.
      python scraper.py --output-dir scraped_data --scraped-data scraped_data/data.json
"""

import argparse
import json
import random
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

SAVE_EVERY = 5  # save to disk every N fetches


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_soup(url: str, session: requests.Session) -> BeautifulSoup:
    resp = session.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def rate_limit(base_delay: float):
    """Sleep for base_delay ± 30 % jitter."""
    jitter = base_delay * 0.3
    time.sleep(base_delay + random.uniform(-jitter, jitter))


def save_json(path: Path, data):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(path)  # atomic-ish rename


def download_image(url: str, output_dir: Path, session: requests.Session) -> str | None:
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
        print(f"    ✗ image download failed {url}: {e}")
        return None


# ---------------------------------------------------------------------------
# Mode 1 – scrape category listing pages
# ---------------------------------------------------------------------------

def get_iconclass_links(session: requests.Session) -> list[dict]:
    soup = get_soup(BROWSE_URL, session)
    browse_list = soup.find("ul", id="browse-list")
    links = (browse_list.find_all("a") if browse_list
             else soup.find_all("a", href=re.compile(r"/browse/illustrations/iconclass/.+")))

    results, seen = [], set()
    for link in links:
        href = link.get("href", "")
        if not href:
            continue
        url = urljoin(BASE_URL, href)
        if url in seen:
            continue
        seen.add(url)
        text = link.get_text(strip=True)
        count_m = re.search(r"\[(\d+)\]", text)
        count = int(count_m.group(1)) if count_m else 0
        code = re.sub(r"\[\d+\]", "", text).strip().rstrip("*")
        results.append({"code": code, "url": url, "count": count})

    print(f"Found {len(results)} iconclass categories")
    return results


def parse_document_part(part, category_code: str) -> dict | None:
    """Parse a single document-part div from a category listing page."""
    data = {"iconclass_bibliasacra_category": category_code}
    table = part.find("table", class_="listTable")
    if not table:
        return None

    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 2:
            continue
        label = cells[0].get_text(strip=True).lower()

        if "illustration id" in label:
            head = row.find("td", class_="listHead")
            if head:
                a = head.find("a")
                data["illustration_id"] = (a.get_text(strip=True) if a
                                           else head.get_text(strip=True))
                if a:
                    data["detail_url"] = urljoin(BASE_URL, a.get("href", ""))

            img_cell = row.find("td", class_="imghead")
            if img_cell:
                img = img_cell.find("img")
                if img:
                    src = img.get("src", "")
                    thumb = urljoin(BASE_URL, src)
                    data["image_thumb_url"] = thumb
                    data["image_url"] = thumb.replace("/thumbs/", "/masters/")
                    m = re.search(r"/(\d+)\.jpg", src)
                    if m:
                        data["image_id"] = m.group(1)
                if img and img.get("alt"):
                    data["reference"] = img.get("alt")

        elif "short title" in label:
            v = row.find("td", class_="listValue")
            data["short_title"] = (v or cells[1]).get_text(strip=True)
        elif label == "type":
            v = row.find("td", class_="listValue")
            data["type"] = (v or cells[1]).get_text(strip=True)
        elif "material" in label:
            v = row.find("td", class_="listValue")
            data["material"] = (v or cells[1]).get_text(strip=True)
        elif "artist" in label:
            v = row.find("td", class_="listValue")
            data["artist"] = (v or cells[1]).get_text(strip=True)

    return data if "illustration_id" in data else None


def scrape_category_page(url: str, code: str, session: requests.Session,
                         delay: float) -> list[dict]:
    visited, queue, results = set(), [url], []
    while queue:
        cur = queue.pop(0)
        if cur in visited:
            continue
        visited.add(cur)
        try:
            soup = get_soup(cur, session)
        except Exception as e:
            print(f"    ✗ {cur}: {e}")
            continue
        for part in soup.find_all("div", class_="document-part"):
            ill = parse_document_part(part, code)
            if ill:
                results.append(ill)
        # pagination
        pag = soup.find("ul", class_="pagination") or soup.find("nav", class_="pagination")
        if pag:
            for a in pag.find_all("a", href=True):
                pu = urljoin(BASE_URL, a["href"])
                if pu not in visited:
                    queue.append(pu)
        rate_limit(delay)
    return results


def run_mode1(args):
    """Mode 1: scrape category listing pages → base dataset."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_file = output_dir / "data.json"
    session = requests.Session()

    # Resume support
    all_data: list[dict] = []
    scraped_codes: set[str] = set()
    if data_file.exists():
        with open(data_file) as f:
            all_data = json.load(f)
        scraped_codes = {d["iconclass_bibliasacra_category"] for d in all_data}
        print(f"Resuming: {len(all_data)} illustrations from "
              f"{len(scraped_codes)} categories already on disk")

    print("Fetching category index …")
    categories = get_iconclass_links(session)
    rate_limit(args.delay)

    if args.limit > 0:
        categories = categories[:args.limit]

    total = len(categories)
    unsaved = 0
    for i, cat in enumerate(categories, 1):
        code = cat["code"]
        if code in scraped_codes:
            print(f"[{i}/{total}] skip {code} (done)")
            continue

        print(f"[{i}/{total}] {code}  (≈{cat['count']} items) …")
        ills = scrape_category_page(cat["url"], code, session, args.delay)
        all_data.extend(ills)
        unsaved += 1
        print(f"    → {len(ills)} illustrations")

        if unsaved >= SAVE_EVERY:
            save_json(data_file, all_data)
            unsaved = 0
            print(f"    💾 saved ({len(all_data)} total)")

    save_json(data_file, all_data)
    print(f"\n✔  Mode 1 done – {len(all_data)} illustrations → {data_file}")


# ---------------------------------------------------------------------------
# Mode 2 – fetch illustration detail pages + download images
# ---------------------------------------------------------------------------

def fetch_illustration_details(ill: dict, session: requests.Session) -> dict:
    """
    Fetch /illustration/<id> and enrich the dict with:
      - iconclass_codes        (list, e.g. ["49M431","11D3261",...])
      - iconclass_description  (full text)
      - detail_title           (full untruncated title)
      - detail_sizes
      - detail_colour
      - detail_oldest_appearance
      - detail_appearances     (list of {image_id, scan_label, edition_id})
    """
    ill_id = ill.get("illustration_id", "")
    if not ill_id:
        return ill

    url = ill.get("detail_url") or f"{BASE_URL}/illustration/{ill_id}"
    try:
        soup = get_soup(url, session)
    except Exception as e:
        ill["_detail_error"] = str(e)
        return ill

    table = soup.find("table", class_="listTable")
    if not table:
        ill["_detail_error"] = "no table found"
        return ill

    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 2:
            continue
        label = cells[0].get_text(strip=True)
        value = cells[1].get_text(strip=True)
        ll = label.lower()

        if ll == "title":
            ill["detail_title"] = value
        elif ll == "type":
            ill["detail_type"] = value
        elif ll == "sizes":
            ill["detail_sizes"] = value
        elif ll == "material":
            ill["detail_material"] = value
        elif ll == "colour":
            ill["detail_colour"] = value
        elif "oldest appearance" in ll:
            ill["detail_oldest_appearance"] = value
        elif ll in ("iconclass", "ornamentcode"):
            ill["iconclass_codes"] = [c.strip() for c in value.split(",") if c.strip()]
        elif ll == "iconclass description":
            ill["iconclass_description"] = value

    # Second listTable → image appearances
    tables = soup.find_all("table", class_="listTable")
    if len(tables) >= 2:
        appearances = []
        for cell in tables[1].find_all("td"):
            links = cell.find_all("a", href=True)
            entry: dict = {}
            for a in links:
                href = a["href"]
                text = a.get_text(strip=True)
                if href.startswith("/image/"):
                    entry["image_id"] = href.split("/image/")[-1]
                    if text:
                        entry["scan_label"] = text
                elif href.startswith("/edition/"):
                    entry["edition_id"] = href.split("/edition/")[-1]
            if entry:
                appearances.append(entry)
        # if appearances:
        #     ill["detail_appearances"] = appearances

    ill["_detail_fetched"] = True
    return ill


def run_mode2(args):
    """Mode 2: enrich existing data with detail pages + download images."""
    data_file = Path(args.scraped_data)
    if not data_file.exists():
        print(f"✗ File not found: {data_file}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    with open(data_file) as f:
        all_data: list[dict] = json.load(f)

    session = requests.Session()
    pending = [i for i, d in enumerate(all_data) if not d.get("_detail_fetched")]
    total = len(pending)
    print(f"Mode 2: {total} illustrations need detail fetching "
          f"({len(all_data) - total} already done)")

    unsaved = 0
    for seq, idx in enumerate(pending, 1):
        ill = all_data[idx]
        ill_id = ill.get("illustration_id", "?")
        print(f"[{seq}/{total}] {ill_id[:80]} …", end=" ", flush=True)

        fetch_illustration_details(ill, session)

        # Download master image
        img_url = ill.get("image_url")
        if img_url:
            fname = download_image(img_url, images_dir, session)
            if fname:
                ill["image_file"] = fname

        err = ill.get("_detail_error")
        if err:
            print(f"⚠ {err}")
        else:
            codes = ill.get("iconclass_codes", [])
            print(f"✓ iconclass={','.join(codes[:3])}{'…' if len(codes) > 3 else ''}")

        unsaved += 1
        if unsaved >= SAVE_EVERY:
            save_json(data_file, all_data)
            unsaved = 0
            print(f"    💾 saved")

        rate_limit(args.delay)

    save_json(data_file, all_data)
    print(f"\n✔  Mode 2 done – enriched {total} illustrations, saved → {data_file}")
    print(f"   Images → {images_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Scrape bibliasacra.nl illustrations (two-pass)")
    parser.add_argument("--output-dir", default="scraped_data",
                        help="Output directory (default: scraped_data)")
    parser.add_argument("--delay", type=float, default=5.0,
                        help="Base delay between requests in seconds (default: 3.0)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit categories to scrape in mode 1 (0 = all)")
    parser.add_argument("--scraped-data", default=None,
                        help="Path to existing data.json → activates mode 2 "
                             "(fetch illustration details + images)")
    args = parser.parse_args()

    if args.scraped_data:
        run_mode2(args)
    else:
        run_mode1(args)


if __name__ == "__main__":
    main()
