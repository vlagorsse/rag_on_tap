import argparse
import logging
import os
import random

import pandas as pd

logger = logging.getLogger(__name__)


def enrich_with_reproducible_entropy(df):
    """
    Enriches the dataframe with a reproducible entropy column 'enriched_story'
    based on the beer style.
    """
    # Set the seed at the start for global reproducibility
    random.seed(0)

    # 1. Expanded mapping with 10 variations per style
    enrichment_library = {
        "ipa": {
            "flavors": [
                "Hoppy with intense notes of grapefruit and pine resin.",
                "Bursting with tropical mango and passionfruit aromas.",
                "Classic dank bitterness with a floral, earthy finish.",
                "Citrus-forward with bright lemon zest and orange peel.",
                "Herbal and spicy with a sharp, lingering hop bite.",
                "Juicy stone fruit notes with a soft, hazy mouthfeel.",
                "Resinous cedar and spruce tips with a dry finish.",
                "Melon and apricot sweetness balanced by high alpha acids.",
                "Bold pineapple and coconut 'Sabro' style hop profile.",
                "Old-school caramel malt backbone with assertive bitterness.",
            ],
            "pairings": [
                "Spicy Thai curry",
                "Blue cheese burger",
                "Buffalo wings",
                "Carrot cake",
                "Fish tacos",
                "Sharp cheddar",
                "Aged gouda",
                "Spicy tuna rolls",
                "Grilled lamb chops",
                "Fried chicken",
            ],
            "difficulty": ["Medium", "Hard"],
        },
        "stout": {
            "flavors": [
                "Velvety chocolate with a strong espresso finish.",
                "Deeply roasted malts with hints of charred oak.",
                "Creamy oats and milk sugar with a silky mouthfeel.",
                "Dry Irish style with a clean, coffee-like astringency.",
                "Toasted marshmallows and vanilla bean sweetness.",
                "Smoky tobacco and leather with a dark fruit undertone.",
                "Rich molasses and black licorice complexity.",
                "Nutty hazelnut and praline notes.",
                "Imperial strength with warming alcohol and dark cherry.",
                "Burnt toast and unsweetened cocoa powder.",
            ],
            "pairings": [
                "Oysters",
                "Chocolate lava cake",
                "Smoked brisket",
                "Vanilla ice cream",
                "Stilton cheese",
                "Beef stew",
                "Grilled venison",
                "Truffle fries",
                "Bread pudding",
                "Blackberry tart",
            ],
            "difficulty": ["Medium", "Hard"],
        },
        "cream ale": {
            "flavors": [
                "Ultra-crisp with a faint hint of sweet corn.",
                "Clean bready notes with a very fast finish.",
                "Subtle floral hops with a light, effervescent body.",
                "Mildly fruity esters with a creamy carbonation.",
                "Pilsner-like crispness but with a rounder mouthfeel.",
                "Biscuity malt with a touch of honey sweetness.",
                "Straw and hay aromas with minimal bitterness.",
                "Low-key pear and apple fruitiness.",
                "Classic 'lawnmower beer' with high crushability.",
                "Sparkling clarity with a short, sweet aftertaste.",
            ],
            "pairings": [
                "Grilled chicken",
                "Lemon-pepper wings",
                "Garden salad",
                "Fish and chips",
                "Monterey Jack",
                "Calamari",
                "Margherita pizza",
                "Light pasta",
                "Popcorn",
                "Summer rolls",
                "Salted pretzels",
            ],
            "difficulty": ["Easy"],
        },
    }

    def get_entropy_row(row):
        style = str(row.get("Style", "")).lower()

        # Determine the category
        category = (
            "cream ale"
            if "cream ale" in style
            else ("ipa" if "ipa" in style else ("stout" if "stout" in style else None))
        )

        if not category:
            return "A balanced beer with classic malt and hop characteristics."

        # Pick a random index (0-9) based on the current state of the seed
        idx = random.randint(0, 9)

        lib = enrichment_library[category]
        flavor = lib["flavors"][idx]
        pairing = lib["pairings"][idx]
        diff = random.choice(lib["difficulty"])

        return f"{flavor} Pairs best with {pairing}. Difficulty: {diff}."

    # 2. Apply to the dataframe
    if "Style" in df.columns:
        df["enriched_story"] = df.apply(get_entropy_row, axis=1)
    return df


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Enrich beer recipe data with descriptive stories."
    )
    parser.add_argument("input", help="Path to the input CSV file")
    parser.add_argument("output", help="Path where the enriched CSV file will be saved")

    args = parser.parse_args()

    if os.path.exists(args.input):
        logger.info(f"Loading data from {args.input}...")
        df = pd.read_csv(args.input, encoding="latin-1", on_bad_lines="skip")
        df = enrich_with_reproducible_entropy(df)

        logger.info(f"Saving enriched data to {args.output}...")
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        df.to_csv(args.output, index=False)
        logger.info("Data enriched and saved successfully.")
    else:
        logger.error(f"Input file '{args.input}' not found.")


if __name__ == "__main__":
    main()
