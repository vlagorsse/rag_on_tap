import argparse
import logging
import os
import random
import re
import time

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.brewersfriend.com"
SEARCH_URL = "https://www.brewersfriend.com/search/index.php"
RECIPES_DIR = "recipes"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "X-Requested-With": "XMLHttpRequest",
    "Origin": "https://www.brewersfriend.com",
    "Referer": "https://www.brewersfriend.com/search/index.php?sort=views",
}

logger = logging.getLogger(__name__)


def safe_request(url, method="GET", data=None, headers=None, max_retries=5):
    retries = 0
    backoff = 2

    while retries < max_retries:
        try:
            if method == "GET":
                response = requests.get(url, headers=headers)
            else:
                response = requests.post(url, headers=headers, data=data)

            if response.status_code == 429:
                wait_time = backoff * (2**retries) + random.uniform(0, 1)
                logger.warning(
                    f"Rate limited (429). Waiting {wait_time:.2f}s before retry {retries + 1}/{max_retries}..."
                )
                time.sleep(wait_time)
                retries += 1
                continue

            response.raise_for_status()
            return response

        except requests.exceptions.RequestException as e:
            if (
                isinstance(e, requests.exceptions.HTTPError)
                and e.response.status_code == 429
            ):
                # Handled above
                continue
            logger.error(f"Request error: {e}")
            retries += 1
            time.sleep(backoff * retries)

    return None


def get_recipe_links():
    links = []
    page = 1

    while len(links) < 1000:
        logger.info(f"Fetching search page {page} (Collected: {len(links)})...")
        payload = {
            "keyword": "",
            "method": "",
            "units": "",
            "style": "",
            "sort": "views",
            "page": str(page),
        }

        response = safe_request(
            SEARCH_URL, method="POST", headers=headers, data=payload
        )
        if not response:
            logger.error("Failed to fetch search page after retries.")
            break

        soup = BeautifulSoup(response.text, "html.parser")
        page_links = []

        # Find all anchor tags
        for a in soup.find_all("a", href=True):
            href = a["href"]
            # Check if it matches recipe pattern
            if "/homebrew/recipe/view/" in href:
                full_url = BASE_URL + href if href.startswith("/") else href
                if full_url not in links and full_url not in page_links:
                    page_links.append(full_url)

        if not page_links:
            logger.info("No more recipes found or empty page.")
            break

        logger.info(f"Found {len(page_links)} recipes on page {page}.")
        links.extend(page_links)
        page += 1

        # Be polite with random delay
        time.sleep(random.uniform(2, 4))

    return links[:1000]


def fetch_and_save_recipe(url):
    # Extract ID from URL
    match = re.search(r"/view/(\d+)/", url)
    if not match:
        logger.error(f"Could not extract ID from {url}")
        return

    recipe_id = match.group(1)
    filename = os.path.join(RECIPES_DIR, f"recipe_{recipe_id}.html")

    if os.path.exists(filename):
        return

    response = safe_request(url, headers=headers)
    if response:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(response.text)
        # Random delay between individual recipe fetches
        time.sleep(random.uniform(1, 3))


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    if not os.path.exists(RECIPES_DIR):
        os.makedirs(RECIPES_DIR)

    logger.info("Starting crawl...")
    links = get_recipe_links()
    logger.info(f"Collected {len(links)} recipe links.")

    logger.info("Fetching individual recipes...")
    for i, link in enumerate(links):
        if (i + 1) % 10 == 0:
            logger.info(f"Progress: {i + 1}/{len(links)}")
        fetch_and_save_recipe(link)

    logger.info("Done.")


if __name__ == "__main__":
    main()
