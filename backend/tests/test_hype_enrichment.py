import json
import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from utilities.hype_enrichment import (create_dynamic_model,
                                       extract_metadata_and_text,
                                       process_recipes)


def test_extract_metadata_and_text(tmp_path):
    # Create a dummy HTML file for testing
    html_content = """
    <h1 itemprop="name">Test Beer</h1>
    <span itemprop="recipeCategory">Test Style</span>
    <div id="view_text_dialog">
        <textarea>
        ABV (standard): 5.5%
        IBU (tinseth): 40.0
        Some recipe details...
        </textarea>
    </div>
    <div class="brewpart">
        <a name="notes"></a>
        <div class="ui message">Test notes.</div>
    </div>
    <table class="bf_recipe_comments">
        <table class="bf_recipe_comment">
            <tr><td>Great beer!</td></tr>
        </table>
    </table>
    """
    recipe_file = tmp_path / "recipe_123.html"
    recipe_file.write_text(html_content, encoding="utf-8")

    data = extract_metadata_and_text(str(recipe_file))

    assert data["BeerID"] == "123"
    assert data["Name"] == "Test Beer"
    assert data["Style"] == "Test Style"
    assert data["ABV"] == "5.5"
    assert data["IBU"] == "40.0"
    assert "Test notes." in data["clean_text"]
    assert "Great beer!" in data["clean_text"]


def test_create_dynamic_model():
    model = create_dynamic_model()
    assert model.__name__ == "BeerAnalysis"
    # Check if some expected fields exist
    assert "appearance_color" in model.model_fields
    assert "overall_impression" in model.model_fields


@patch("utilities.hype_enrichment.ChatGoogleGenerativeAI")
@patch("utilities.hype_enrichment.DataService")
@patch("glob.glob")
@patch("utilities.hype_enrichment.GOOGLE_API_KEY", "fake_key")
def test_process_recipes_flow(
    mock_glob, mock_data_service_class, mock_chat_google_class, tmp_path
):
    # Create the directory and file on disk to avoid FileNotFoundError in os.path.getmtime
    recipes_dir = tmp_path / "recipes"
    recipes_dir.mkdir()
    recipe_file = recipes_dir / "recipe_123.html"
    recipe_file.write_text("dummy", encoding="utf-8")

    # Mock glob to return our temp file
    mock_glob.return_value = [str(recipe_file)]

    # Mock DataService
    mock_data_service = MagicMock()
    mock_data_service.load.return_value = pd.DataFrame()
    mock_data_service_class.return_value = mock_data_service

    # Mock LLM and structured output
    mock_llm = MagicMock()
    mock_structured_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured_llm

    # Mock result from LLM
    mock_result = MagicMock()
    mock_result.model_dump.return_value = {
        "appearance_color": "Golden",
        "appearance_clarity": "Clear",
        "appearance_head": "White",
        "appearance_carbonation": "High",
        "aroma_malt_profile": "Sweet",
        "aroma_hop_aroma": "Floral",
        "aroma_esters_phenols": "None",
        "flavor_balance": "Balanced",
        "flavor_bitterness": "Moderate",
        "flavor_aftertaste": "Clean",
        "mouthfeel_body": "Medium",
        "mouthfeel_carbonation": "Prickly",
        "mouthfeel_warmth": "None",
        "overall_impression": "Good",
        "overall_style_accuracy": "High",
        "overall_beer_clone": "Yes",
        "overall_other_comments": "None",
    }
    mock_structured_llm.invoke.return_value = mock_result

    mock_chat_google_class.return_value = mock_llm

    # Mock extract_metadata_and_text to avoid reading real files
    with patch("utilities.hype_enrichment.extract_metadata_and_text") as mock_extract:
        mock_extract.return_value = {
            "BeerID": "123",
            "Name": "Test",
            "Style": "Test",
            "ABV": "5.0",
            "IBU": "30",
            "clean_text": "Recipe text...",
        }

        output_csv = str(tmp_path / "output.csv")
        process_recipes(output_csv, resume=False)

    # Check that DataService.save was called
    assert mock_data_service.save.called
    args, _ = mock_data_service.save.call_args
    df_saved = args[0]
    assert not df_saved.empty
    assert df_saved.iloc[0]["BeerID"] == "123"
    assert df_saved.iloc[0]["appearance_color"] == "Golden"
    assert "enriched_story" in df_saved.columns
