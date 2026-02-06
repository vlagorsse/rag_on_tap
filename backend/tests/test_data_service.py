import os

import pandas as pd
import pytest

from services.data_service import DataService


class TestDataService:
    @pytest.fixture
    def temp_csv(self, tmp_path):
        """Fixture to create a temporary CSV file path."""
        return str(tmp_path / "test_recipes.csv")

    def test_save_and_load(self, temp_csv):
        """Test that DataService correctly saves and loads a DataFrame."""
        service = DataService(temp_csv)
        df = pd.DataFrame({"BeerID": [1, 2], "Name": ["Beer A", "Beer B"]})

        service.save(df)
        loaded_df = service.load()

        assert not loaded_df.empty
        assert len(loaded_df) == 2
        assert list(loaded_df["Name"]) == ["Beer A", "Beer B"]

    def test_load_non_existent(self):
        """Test that DataService returns an empty DataFrame for non-existent files."""
        service = DataService("non_existent.csv")
        df = service.load()
        assert df.empty

    def test_fallback_encoding(self, tmp_path):
        """Test that DataService falls back to latin-1 on UnicodeDecodeError."""
        csv_path = tmp_path / "latin1.csv"
        # Create a CSV with latin-1 encoding
        df = pd.DataFrame({"Name": ["Bière"]})
        df.to_csv(csv_path, index=False, encoding="latin-1")

        service = DataService(str(csv_path))
        # Should fallback to latin-1 and load correctly
        loaded_df = service.load(encoding="utf-8")

        assert not loaded_df.empty
        assert loaded_df.iloc[0]["Name"] == "Bière"
