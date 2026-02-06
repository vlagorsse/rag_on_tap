import logging

import pandas as pd

logger = logging.getLogger(__name__)


class DataService:
    """Handles reading and writing of recipe data in CSV format."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self, encoding: str = "utf-8") -> pd.DataFrame:
        """Loads the CSV file into a pandas DataFrame with encoding fallback."""
        try:
            return pd.read_csv(self.file_path, encoding=encoding)
        except FileNotFoundError:
            return pd.DataFrame()
        except UnicodeDecodeError:
            logger.warning(
                f"UnicodeDecodeError for {self.file_path}, falling back to latin-1."
            )
            return pd.read_csv(self.file_path, encoding="latin-1")

    def save(self, df: pd.DataFrame):
        """Saves the DataFrame to the CSV file path."""
        df.to_csv(self.file_path, index=False)
