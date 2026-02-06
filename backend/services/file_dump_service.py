import json
import logging

from langchain_core.documents import Document

from services.storage_service import StorageService

logger = logging.getLogger(__name__)


class FileDumpService(StorageService):
    """Service that dumps documents to a file instead of a vector store."""

    def __init__(self, output_path: str):
        self.output_path = output_path

    def add_documents(self, documents: list[Document], batch_size: int = 500):
        """Dumps documents to a JSONL file."""
        logger.info(f"Dumping {len(documents)} documents to {self.output_path}...")

        with open(self.output_path, "w", encoding="utf-8") as f:
            for doc in documents:
                # Convert Document to a serializable dictionary
                dump_data = {"page_content": doc.page_content, "metadata": doc.metadata}
                f.write(json.dumps(dump_data) + "\n")

        logger.info(f"Successfully dumped documents to {self.output_path}")
