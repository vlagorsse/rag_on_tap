import argparse
import logging
import os
import random
import re
import time

from bs4 import BeautifulSoup
from fetch_utils import fetch_and_save_recipe, safe_request, setup_logging

BASE_URL = "https://beersmithrecipes.com"
START_URL = "https://beersmithrecipes.com/mostcommented"
RECIPES_DIR = "recipes/beer_smith"
ID_PATTERN = r"/viewrecipe/(\d+)/"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
}

logger = logging.getLogger(__name__)


def get_recipe_links(max_pages=10):
    links = []

    for i in range(max_pages):
        if i == 0:
            url = START_URL
        else:
            url = f"{START_URL}/{i}"

        logger.info(f"Fetching search page {i+1}: {url} (Collected: {len(links)})...")

        response = safe_request(url, headers=headers)
        if not response:
            logger.error(f"Failed to fetch page {i+1}.")
            break

        soup = BeautifulSoup(response.text, "html.parser")
        page_links = []

        # Find all recipe links
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/viewrecipe/" in href:
                if href not in links and href not in page_links:
                    page_links.append(href)

        if not page_links:
            logger.info("No more recipes found or empty page.")
            break

        logger.info(f"Found {len(page_links)} recipes on page {i+1}.")
        links.extend(page_links)

        # Be polite
        time.sleep(random.uniform(2, 4))

    return links


def main():
    setup_logging()
    if not os.path.exists(RECIPES_DIR):
        os.makedirs(RECIPES_DIR)

    logger.info("Starting BeerSmith crawl...")
    links = get_recipe_links(max_pages=10)
    logger.info(f"Collected {len(links)} recipe links.")

    logger.info("Fetching individual recipes...")
    for i, link in enumerate(links):
        if (i + 1) % 10 == 0:
            logger.info(f"Progress: {i + 1}/{len(links)}")
        fetch_and_save_recipe(link, RECIPES_DIR, ID_PATTERN, headers=headers)

    logger.info("Done.")


if __name__ == "__main__":
    main()
