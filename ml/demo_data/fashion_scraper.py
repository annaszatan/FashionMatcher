"""
Fashion Retailer Scraper - Created by Claude AI
========================
Scrapes clothing images and metadata from RW&Co, American Eagle, and Old Navy.

Output structure (mirrors DeepFashion2):
    dataset/
    ├── images/
    │   ├── rwco/
    │   │   └── rwco_0001.jpg
    │   ├── american_eagle/
    │   │   └── ae_0001.jpg
    │   └── old_navy/
    │       └── on_0001.jpg
    └── annos/
        ├── rwco/
        │   └── rwco_0001.json
        ├── american_eagle/
        │   └── ae_0001.json
        └── old_navy/
            └── on_0001.json

Each annotation JSON contains:
    {
        "id": "rwco_0001",
        "source": "rwco",
        "product_name": "...",
        "product_url": "...",
        "image_url": "...",
        "image_file": "images/rwco/rwco_0001.jpg",
        "price": "...",
        "category": "...",
        "color": "...",
        "scraped_at": "2024-01-01T00:00:00"
    }

Usage:
    pip install playwright beautifulsoup4 requests
    playwright install chromium

    python fashion_scraper.py                        # scrape all sites
    python fashion_scraper.py --sites rwco ae        # scrape specific sites
    python fashion_scraper.py --max-items 50         # limit items per site
    python fashion_scraper.py --output ./my_dataset  # custom output directory

NOTE:
- This scraper was used for the demo only
- American Eagle should work but the other sites are less likely to work as expected
- Will need to change the first link in array to adjust what page to pull from in AE (will not automatically pull from every link)
"""

import os
import re
import json
import time
import random
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
from urllib.parse import urljoin, urlparse
import io
from PIL import Image

import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("scraper.log"),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def slugify(text: str) -> str:
    """Convert text to a safe filename slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "_", text)
    return text[:60]


def random_delay(min_s: float = 1.5, max_s: float = 4.0):
    """Polite random delay between requests."""
    time.sleep(random.uniform(min_s, max_s))


def save_image(image_url: str, dest_path: Path, session: requests.Session) -> bool:
    """Download image, skip small icons / placeholders."""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Referer": image_url,
        }
        resp = session.get(image_url, headers=headers, timeout=15, stream=True)
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")
        if "image" not in content_type and not image_url.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        ):
            log.warning(f"Skipping non-image URL: {image_url}")
            return False

        # Read image in memory to check size
        img_data = resp.content
        try:
            img = Image.open(io.BytesIO(img_data))
            if img.width < 100 or img.height < 100:
                log.warning(f"Skipping small image (likely icon): {image_url}")
                return False
        except Exception:
            log.warning(f"Cannot open image (skipping): {image_url}")
            return False

        # Save to disk
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "wb") as f:
            f.write(img_data)

        return True

    except Exception as e:
        log.error(f"Failed to download {image_url}: {e}")
        return False


def save_annotation(anno: dict, dest_path: Path):
    """Save a product annotation as JSON."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "w", encoding="utf-8") as f:
        json.dump(anno, f, indent=2, ensure_ascii=False)


TROUSER_KEYWORDS = {"pant", "pants", "trouser", "trousers", "bottoms", "chino", "chinos"}
TSHIRT_KEYWORDS  = {"t-shirt", "tshirt", "tee", "tees", "graphic-tee", "graphic-tees"}

def infer_label(url: str, category: str) -> str:
    """Return 'tshirt', 'trousers', or 'unknown' based on URL / category string."""
    combined = (url + " " + category).lower()
    if any(kw in combined for kw in TSHIRT_KEYWORDS):
        return "tshirt"
    if any(kw in combined for kw in TROUSER_KEYWORDS):
        return "trousers"
    return "unknown"

def build_dataset_index(output_dir: Path):
    """
    Walk all annotation files and write a single index.json —
    useful for loading the whole dataset at once in your model pipeline.
    """
    index = []
    for anno_file in sorted((output_dir / "annos").rglob("*.json")):
        with open(anno_file, encoding="utf-8") as f:
            index.append(json.load(f))

    index_path = output_dir / "index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    log.info(f"Index written → {index_path} ({len(index)} items)")


# ---------------------------------------------------------------------------
# Base scraper
# ---------------------------------------------------------------------------

class BaseScraper:
    """
    Common interface for all retailer scrapers.

    Subclasses must implement:
        - get_product_urls(page, category_url) -> list[str]
        - get_category_urls() -> list[str]
        - parse_product(page, url) -> Optional[dict]
    """

    source_key: str = ""          # e.g. "rwco"
    prefix: str = ""              # filename prefix, e.g. "rwco"
    base_url: str = ""

    def __init__(self, output_dir: Path, max_items: int = 200):
        self.output_dir = output_dir
        self.max_items = max_items
        self.session = requests.Session()
        
        # Find existing items to continue numbering
        anno_dir = self.output_dir / "annos" / self.source_key
        if anno_dir.exists():
            existing = list(anno_dir.glob(f"{self.prefix}_*.json"))
            if existing:
                nums = [int(f.stem.split("_")[-1]) for f in existing]
                self._counter = max(nums)
            else:
                self._counter = 0
        else:
            self._counter = 0

    # -- Internal helpers ----------------------------------------------------

    def _img_path(self, item_id: str, ext: str = "jpg") -> Path:
        return self.output_dir / "images" / self.source_key / f"{item_id}.{ext}"

    def _anno_path(self, item_id: str) -> Path:
        return self.output_dir / "annos" / self.source_key / f"{item_id}.json"

    def _next_id(self) -> str:
        self._counter += 1
        return f"{self.prefix}_{self._counter:04d}"

    def _already_scraped(self, item_id: str) -> bool:
        return self._anno_path(item_id).exists()

    # -- Public entry point --------------------------------------------------

    def run(self, playwright):
        browser = playwright.chromium.launch(headless=False, slow_mo=100)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1440, "height": 900},
            locale="en-US",
            java_script_enabled=True,
        )
        page = context.new_page()

        scraped = 0
        try:
            for cat_url in self.get_category_urls():
                if scraped >= self.max_items:
                    break
                log.info(f"[{self.source_key}] Category → {cat_url}")

                try:
                    product_urls = self.get_product_urls(page, cat_url)
                except Exception as e:
                    log.error(f"[{self.source_key}] Failed to get product URLs: {e}")
                    continue

                for url in product_urls:
                    if scraped >= self.max_items:
                        break
                    try:
                        item_id = self._next_id()
                        product = self.parse_product(page, url)
                        if not product:
                            continue

                        # Download image
                        img_ext = "jpg"
                        if product.get("image_url", "").endswith(".png"):
                            img_ext = "png"
                        img_path = self._img_path(item_id, img_ext)
                        img_ok = save_image(product["image_url"], img_path, self.session)
                        if not img_ok:
                            continue

                        # Build annotation
                        anno = {
                            "id": item_id,
                            "source": self.source_key,
                            "label": infer_label(url, product.get("category", "")),
                            "product_name": product.get("product_name", ""),
                            "product_url": url,
                            "image_url": product.get("image_url", ""),
                            "image_file": str(img_path.relative_to(self.output_dir)),
                            "price": product.get("price", ""),
                            "category": product.get("category", ""),
                            "color": product.get("color", ""),
                            "description": product.get("description", ""),
                            "scraped_at": datetime.utcnow().isoformat(),
                        }
                        save_annotation(anno, self._anno_path(item_id))
                        scraped += 1
                        log.info(f"[{self.source_key}] ✓ {item_id} — {anno['product_name'][:50]}")
                        random_delay()

                    except Exception as e:
                        log.error(f"[{self.source_key}] Error on {url}: {e}")
                        continue

        finally:
            context.close()
            browser.close()

        log.info(f"[{self.source_key}] Done. {scraped} items saved.")
        return scraped

    # -- Subclass interface --------------------------------------------------

    def get_category_urls(self) -> list:
        raise NotImplementedError

    def get_product_urls(self, page, category_url: str) -> list:
        raise NotImplementedError

    def parse_product(self, page, url: str) -> Optional[dict]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# RW&Co scraper
# ---------------------------------------------------------------------------

class RWCoScraper(BaseScraper):
    source_key = "rwco"
    prefix = "rwco"
    base_url = "https://www.rwandco.com"

    CATEGORIES = [
        # T-shirts
        "https://www.rwandco.com/en/women/tops/t-shirts",
        "https://www.rwandco.com/en/men/tops/t-shirts",
        # Trousers / pants
        "https://www.rwandco.com/en/women/bottoms/pants",
        "https://www.rwandco.com/en/men/pants",
    ]

    def get_category_urls(self):
        return self.CATEGORIES

    def get_product_urls(self, page, category_url: str) -> list:
        page.goto(category_url, wait_until="domcontentloaded", timeout=30000)
        page.wait_for_timeout(3000)

        # Scroll to trigger lazy-loading
        for _ in range(3):
            page.evaluate("window.scrollBy(0, window.innerHeight)")
            page.wait_for_timeout(1500)

        html = page.content()
        soup = BeautifulSoup(html, "html.parser")

        links = set()
        for a in soup.select("a[href]"):
            href = a["href"]
            if "/en/women/" in href or "/en/men/" in href:
                full = urljoin(self.base_url, href)
                # Only product detail pages (have a product-name segment after category)
                parts = [p for p in urlparse(full).path.split("/") if p]
                if len(parts) >= 4:
                    links.add(full)

        return list(links)[:50]

    def parse_product(self, page, url: str) -> Optional[dict]:
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        page.wait_for_timeout(2000)
        html = page.content()
        soup = BeautifulSoup(html, "html.parser")

        # Product name
        name_el = soup.select_one("h1.product-name, h1[class*='product'], h1")
        product_name = name_el.get_text(strip=True) if name_el else ""

        # Price
        price_el = soup.select_one(
            "[class*='price'] [class*='sale'], [class*='price']"
        )
        price = price_el.get_text(strip=True) if price_el else ""

        # Main image — prefer the high-res src or srcset
        img_el = soup.select_one(
            "img[class*='product'], .product-image img, [class*='gallery'] img"
        )
        image_url = ""
        if img_el:
            image_url = (
                img_el.get("data-src")
                or img_el.get("data-zoom-image")
                or img_el.get("src", "")
            )
            if image_url.startswith("//"):
                image_url = "https:" + image_url
            elif image_url.startswith("/"):
                image_url = urljoin(self.base_url, image_url)

        if not image_url or not product_name:
            return None

        # Category from URL path
        path_parts = urlparse(url).path.strip("/").split("/")
        category = path_parts[2] if len(path_parts) >= 3 else ""

        # Color (often in page title or variant selector)
        color_el = soup.select_one(
            "[class*='color-name'], [class*='swatch-label'], [aria-label*='color']"
        )
        color = color_el.get_text(strip=True) if color_el else ""

        # Description
        desc_el = soup.select_one(
            "[class*='product-description'], [class*='pdp-description']"
        )
        description = desc_el.get_text(" ", strip=True)[:500] if desc_el else ""

        return {
            "product_name": product_name,
            "image_url": image_url,
            "price": price,
            "category": category,
            "color": color,
            "description": description,
        }


# ---------------------------------------------------------------------------
# American Eagle scraper
# ---------------------------------------------------------------------------

class AmericanEagleScraper(BaseScraper):
    source_key = "american_eagle"
    prefix = "ae"
    base_url = "https://www.ae.com"

    CATEGORIES = [
        # Bottoms
        "https://www.ae.com/ca/en/c/women/bottoms/cat10051",
        "https://www.ae.com/ca/en/c/men/bottoms/cat10027",
        # Shirts / T-Shirts
        "https://www.ae.com/us/en/c/men/tops/t-shirts/cat90012",
        "https://www.ae.com/us/en/c/women/tops/t-shirts/cat90030",
    ]

    def get_category_urls(self):
        return self.CATEGORIES

    def get_product_urls(self, page, category_url: str) -> list:
        page.goto(category_url, timeout=60000)

        # Wait for ANY anchor (simpler + more reliable)
        page.wait_for_selector("a", timeout=15000)

        # Give time for JS rendering
        page.wait_for_timeout(5000)

        # Scroll to trigger lazy loading
        for _ in range(10):
            page.mouse.wheel(0, 5000)
            page.wait_for_timeout(1000)

        # Extract links
        links = set()
        elements = page.query_selector_all("a")

        for el in elements:
            href = el.get_attribute("href")
            if href and "/p/" in href:
                links.add(urljoin(self.base_url, href.split("?")[0]))

        print(f"[AE] Found {len(links)} product links")

        # DEBUG SAVE
        with open("ae_debug.html", "w", encoding="utf-8") as f:
            f.write(page.content())

        return list(links)[:self.max_items]

    def parse_product(self, page, url: str) -> Optional[dict]:
        try:
            page.goto(url, timeout=60000)

            # Wait for product title (key signal page loaded)
            page.wait_for_selector("h1", timeout=15000)

            # --- NAME ---
            name_el = page.query_selector("h1")
            product_name = name_el.inner_text().strip() if name_el else ""

            # --- PRICE ---
            price_el = page.query_selector("[class*='price']")
            price = price_el.inner_text().strip() if price_el else ""

            # --- IMAGE ---
            img_el = page.query_selector("img[src*='scene7']")
            image_url = ""

            if img_el:
                image_url = img_el.get_attribute("src") or ""

                # force higher resolution
                if "?" in image_url:
                    image_url = image_url.split("?")[0] + "?$pdp-main-desktop$"

                if image_url.startswith("//"):
                    image_url = "https:" + image_url

            # --- COLOR ---
            color_el = page.query_selector("[class*='color'], [aria-label*='color']")
            color = color_el.inner_text().strip() if color_el else ""

            # --- DESCRIPTION ---
            desc_el = page.query_selector("[class*='description']")
            description = desc_el.inner_text().strip()[:300] if desc_el else ""

            if not product_name or not image_url:
                print(f"[AE] Skipping (missing data): {url}")
                return None

            return {
                "product_name": product_name,
                "image_url": image_url,
                "price": price,
                "category": "tshirt",
                "color": color,
                "description": description,
            }

        except Exception as e:
            print(f"[AE] Failed parsing {url}: {e}")
            return None


# ---------------------------------------------------------------------------
# Old Navy scraper
# ---------------------------------------------------------------------------

class OldNavyScraper(BaseScraper):
    source_key = "old_navy"
    prefix = "on"
    base_url = "https://oldnavy.gap.com"

    CATEGORIES = [
        # T-shirts (women's & men's)
        "https://oldnavy.gap.com/browse/category.do?cid=1159639",   # Women's tops (filter to tees)
        "https://oldnavy.gap.com/browse/category.do?cid=1159619",   # Men's tops (filter to tees)
        # Trousers / pants
        "https://oldnavy.gap.com/browse/category.do?cid=1159641",   # Women's pants
        "https://oldnavy.gap.com/browse/category.do?cid=1159622",   # Men's pants
    ]

    def get_category_urls(self):
        return self.CATEGORIES

    def get_product_urls(self, page, category_url: str) -> list:
        page.goto(category_url, wait_until="networkidle", timeout=40000)
        page.wait_for_timeout(3000)

        for _ in range(4):
            page.evaluate("window.scrollBy(0, window.innerHeight)")
            page.wait_for_timeout(1500)

        html = page.content()
        soup = BeautifulSoup(html, "html.parser")

        links = set()
        for a in soup.select("a[href]"):
            href = a.get("href", "")
            if "/products/" in href or "/browse/product" in href:
                full = urljoin(self.base_url, href.split("?")[0])
                links.add(full)

        return list(links)[:50]

    def parse_product(self, page, url: str) -> Optional[dict]:
        page.goto(url, wait_until="networkidle", timeout=40000)
        page.wait_for_timeout(2000)
        html = page.content()
        soup = BeautifulSoup(html, "html.parser")

        # Product name
        name_el = soup.select_one(
            "h1[class*='product-name'], h1[itemprop='name'], h1"
        )
        product_name = name_el.get_text(strip=True) if name_el else ""

        # Price — Old Navy often has sale + original
        price_el = soup.select_one(
            "[class*='sale-price'], [class*='product-price'], [itemprop='price']"
        )
        price = price_el.get_text(strip=True) if price_el else ""

        # Image
        img_el = soup.select_one(
            "[class*='product-image'] img, [class*='main-image'] img, img[class*='product']"
        )
        if not img_el:
            img_el = soup.select_one("img[src*='oldnavy']")
        if not img_el:
            img_el = soup.select_one("img[src*='gap.com']")

        image_url = ""
        if img_el:
            image_url = (
                img_el.get("data-src")
                or img_el.get("src", "")
            )
            if image_url.startswith("//"):
                image_url = "https:" + image_url

        if not image_url or not product_name:
            return None

        # Category
        color_el = soup.select_one(
            "[class*='selected-color'], [class*='color-chip-label']"
        )
        color = color_el.get_text(strip=True) if color_el else ""

        desc_el = soup.select_one(
            "[class*='product-description'], [itemprop='description']"
        )
        description = desc_el.get_text(" ", strip=True)[:500] if desc_el else ""

        category_el = soup.select_one("li.breadcrumb:last-child, [class*='breadcrumb']:last-child")
        category = category_el.get_text(strip=True) if category_el else ""

        return {
            "product_name": product_name,
            "image_url": image_url,
            "price": price,
            "category": category,
            "color": color,
            "description": description,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

SCRAPERS = {
    "rwco": RWCoScraper,
    "ae": AmericanEagleScraper,
    "on": OldNavyScraper,
}


def parse_args():
    p = argparse.ArgumentParser(description="Fashion retailer scraper")
    p.add_argument(
        "--sites",
        nargs="+",
        choices=list(SCRAPERS.keys()),
        default=list(SCRAPERS.keys()),
        help="Which sites to scrape (default: all)",
    )
    p.add_argument(
        "--max-items",
        type=int,
        default=200,
        help="Max items to scrape per site (default: 200)",
    )
    p.add_argument(
        "--output",
        type=str,
        default="./dataset",
        help="Output directory (default: ./dataset)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Output directory: {output_dir.resolve()}")
    log.info(f"Sites: {args.sites}")
    log.info(f"Max items per site: {args.max_items}")

    total = 0
    with sync_playwright() as pw:
        for site_key in args.sites:
            scraper_cls = SCRAPERS[site_key]
            scraper = scraper_cls(output_dir=output_dir, max_items=args.max_items)
            count = scraper.run(pw)
            total += count

    # Build a unified index.json across all scraped items
    build_dataset_index(output_dir)
    log.info(f"All done. Total items scraped: {total}")


if __name__ == "__main__":
    main()