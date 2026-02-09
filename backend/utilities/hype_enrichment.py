import argparse
import datetime
import glob
import json
import logging
import os
import re

import pandas as pd
from bs4 import BeautifulSoup
from dotenv import find_dotenv, load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import Field, create_model

from services.config_service import ConfigService
from services.data_service import DataService

# Load environment variables
load_dotenv(find_dotenv())

# Configuration
MODEL_NAME = "gemini-2.5-flash-lite"
RECIPES_DIR = "recipes"
MAX_RECIPES = 100
QUESTIONS_FILE = "hype_questions.json"
OUTPUT_LOG_FILE = "enrichment_log.txt"

logger = logging.getLogger(__name__)


def load_llm(config: ConfigService):
    # Load config locally to avoid side effects during import
    google_api_key = config.google_api_key

    logger.info(f"Loading model: {MODEL_NAME} via Google AI Studio...")
    if not google_api_key:
        logger.error("GOOGLE_API_KEY not found in environment")
        return None
    try:
        llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key=google_api_key,
            temperature=0.1,
            max_output_tokens=2000,
        )
        return llm
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None


def extract_metadata_and_text(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

        try:
            beer_id = os.path.basename(filepath).split("_")[1].split(".")[0]
        except:
            beer_id = "Unknown"

        try:
            name = soup.find("h1", itemprop="name").get_text(strip=True)
        except:
            name = "Unknown"

        try:
            style = soup.find("span", itemprop="recipeCategory").get_text(strip=True)
        except:
            style = "Unknown"

        text_div = soup.find("div", id="view_text_dialog")
        if text_div:
            textarea = text_div.find("textarea")
            clean_text = (
                textarea.get_text(separator="\n", strip=True)
                if textarea
                else text_div.get_text(separator="\n", strip=True)
            )
        else:
            return None

        notes_anchor = soup.find("a", attrs={"name": "notes"})
        if notes_anchor:
            parent_div = notes_anchor.find_parent("div", class_="brewpart")
            if parent_div:
                notes_content = parent_div.find("div", class_="ui message")
                if notes_content:
                    clean_text += "\n\n--- NOTES ---\n" + notes_content.get_text(
                        separator="\n", strip=True
                    )

        comments_table = soup.find("table", class_="bf_recipe_comments")
        if comments_table:
            clean_text += "\n\n--- REVIEWS & COMMENTS ---\n"
            comment_rows = comments_table.find_all("table", class_="bf_recipe_comment")
            if comment_rows:
                for row in comment_rows:
                    text = " ".join(row.get_text(separator=" ", strip=True).split())
                    clean_text += "- " + text + "\n"
            else:
                clean_text += comments_table.get_text(separator="\n", strip=True)

        abv_match = re.search(r"ABV \(standard\):\s*([\d\.]+)\%", clean_text)
        abv = abv_match.group(1) if abv_match else ""

        ibu_match = re.search(r"IBU \(tinseth\):\s*([\d\.]+)", clean_text)
        ibu = ibu_match.group(1) if ibu_match else ""

        return {
            "BeerID": beer_id,
            "Name": name,
            "Style": style,
            "ABV": abv,
            "IBU": ibu,
            "clean_text": clean_text,
        }


def create_dynamic_model():
    with open(QUESTIONS_FILE, "r") as f:
        questions_data = json.load(f)

    fields = {}
    for category in questions_data["questions"]:
        for item in category["items"]:
            field_name = item["field"]
            description = item["question"]
            fields[field_name] = (str, Field(description=description))

    BeerAnalysis = create_model("BeerAnalysis", **fields)
    return BeerAnalysis


def process_recipes(output_csv, config: ConfigService, resume=False):
    recipe_files = glob.glob(os.path.join(RECIPES_DIR, "*.html"))
    recipe_files.sort(key=os.path.getmtime)
    selected_files = recipe_files[:MAX_RECIPES]

    data_service = DataService(output_csv)
    processed_ids = set()
    results = []
    log_mode = "w"

    if resume and os.path.exists(output_csv):
        logger.info(f"Resuming from {output_csv}...")
        df_existing = data_service.load()
        if not df_existing.empty:
            processed_ids = set(df_existing["BeerID"].astype(str).tolist())
            results = df_existing.to_dict("records")
            logger.info(f"Found {len(processed_ids)} already processed recipes.")
            log_mode = "a"

    llm = load_llm(config)
    if not llm:
        return

    BeerAnalysisModel = create_dynamic_model()
    # Use with_structured_output for reliable JSON
    structured_llm = llm.with_structured_output(BeerAnalysisModel)

    with open(OUTPUT_LOG_FILE, log_mode, encoding="utf-8") as log_f:
        if log_mode == "w":
            log_f.write(
                f"Enrichment Log (Vertex AI - Gemini 2.5 Flash Lite) - Started: {datetime.datetime.now()}\n\n"
            )
        else:
            log_f.write(
                f"\nEnrichment Log (Vertex AI - Gemini 2.5 Flash Lite) - Resumed: {datetime.datetime.now()}\n\n"
            )

        for i, filepath in enumerate(selected_files):
            recipe_name = os.path.basename(filepath)
            data = extract_metadata_and_text(filepath)

            if not data or data["BeerID"] in processed_ids:
                continue

            logger.info(f"Processing recipe {i+1}/{len(selected_files)}: {recipe_name}")
            log_f.write(f"--- Recipe {i+1}/{len(selected_files)}: {recipe_name} ---\n")

            truncated_text = data["clean_text"][:30000]

            prompt = (
                "You are an expert beer sommelier and brewer. Analyze the recipe and provide a structured report.\n"
                "For each field, write at least one or two full, descriptive sentences based on the recipe.\n\n"
                "RECIPE CONTENT:\n"
                f"{truncated_text}"
            )

            try:
                # Invoke with structured output
                structured_data_obj = structured_llm.invoke(prompt)

                # Convert Pydantic model to dict
                structured_data = structured_data_obj.model_dump()

                log_f.write(json.dumps(structured_data, indent=2) + "\n\n")

                story_parts = []
                for field, value in structured_data.items():
                    data[field] = value
                    story_parts.append(f"{field.replace('_', ' ').title()}: {value}")

                data["enriched_story"] = "\n\n".join(story_parts)

                del data["clean_text"]
                results.append(data)
                data_service.save(pd.DataFrame(results))
                logger.info(f"Successfully processed {recipe_name}")

            except Exception as e:
                error_msg = f"Error processing recipe {recipe_name}: {e}"
                logger.error(error_msg)
                log_f.write(error_msg + "\n\n")
            log_f.flush()

    logger.info(f"Enrichment completed. Log saved to {OUTPUT_LOG_FILE}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Enrich beer recipes using Gemini 2.5 Flash Lite with structured output."
    )
    parser.add_argument(
        "--output_csv",
        "-o",
        help="Path to save the enriched data as CSV",
        required=True,
    )
    parser.add_argument(
        "--resume", "-r", action="store_true", help="Resume from existing CSV"
    )
    args = parser.parse_args()

    config = ConfigService()
    process_recipes(args.output_csv, config, args.resume)


if __name__ == "__main__":
    main()
