import argparse
import logging
import os
import random
import re
import time

from bs4 import BeautifulSoup
from fetch_utils import fetch_and_save_recipe, safe_request, setup_logging

BASE_URL = "https://www.brewersfriend.com"
SEARCH_URL = "https://www.brewersfriend.com/search/index.php"
RECIPES_DIR = "recipes/brewers_friend"
ID_PATTERN = r"/view/(\d+)/"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "X-Requested-With": "XMLHttpRequest",
    "Origin": "https://www.brewersfriend.com",
    "Referer": "https://www.brewersfriend.com/search/index.php?sort=views",
}

logger = logging.getLogger(__name__)


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


def main():
    setup_logging()
    if not os.path.exists(RECIPES_DIR):
        os.makedirs(RECIPES_DIR)

    logger.info("Starting crawl...")
    links = get_recipe_links()
    logger.info(f"Collected {len(links)} recipe links.")

    logger.info("Fetching individual recipes...")
    for i, link in enumerate(links):
        if (i + 1) % 10 == 0:
            logger.info(f"Progress: {i + 1}/{len(links)}")
        fetch_and_save_recipe(link, RECIPES_DIR, ID_PATTERN, headers=headers)

    logger.info("Done.")


if __name__ == "__main__":
    main()
