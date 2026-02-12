import logging
import os

import torch
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector

from services.config_service import ConfigService
from services.storage_service import StorageService

logger = logging.getLogger(__name__)


class VectorStoreService(StorageService):
    """Manages the PGVector database connection and operations."""

    def __init__(
        self,
        config: ConfigService,
        model_name: str = "all-MiniLM-L6-v2",
        collection_name: str = "beer_recipes",
        num_threads: int = 2,
    ):
        self.config = config
        self.model_name = model_name
        self.collection_name = collection_name
        self.connection_string = config.connection_string
        self.num_threads = num_threads

        self._initialize_vectorstore()

    def _initialize_vectorstore(self):
        """Initializes embeddings and the PGVector store."""
        logger.info(f"Initializing embedding model ({self.model_name})...")

        # Limit torch threads to avoid making the machine unresponsive on CPU
        if torch.cuda.is_available():
            logger.info("CUDA is available, using GPU.")
            device = "cuda"
        else:
            logger.info(
                f"CUDA not available, using CPU and limiting to {self.num_threads} threads."
            )
            torch.set_num_threads(self.num_threads)
            device = "cpu"

        model_kwargs = {"device": device}
        if "Qwen" in self.model_name:
            model_kwargs["trust_remote_code"] = True

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs=model_kwargs,
        )

        logger.info(f"Connecting to PGVector collection '{self.collection_name}'...")
        self.vectorstore = PGVector(
            embeddings=self.embeddings,
            collection_name=self.collection_name,
            connection=self.connection_string,
            use_jsonb=True,
        )

    def add_documents(self, documents: list[Document], batch_size: int = 100):
        """Adds documents to the vector store in batches."""
        if not documents:
            return

        logger.info(f"Adding {len(documents)} documents to the vector store...")
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            logger.info(
                f"Processing batch {i//batch_size + 1}: documents {i} to {min(i + batch_size, len(documents))}"
            )
            self.vectorstore.add_documents(batch)
            logger.info(
                f"Completed batch {i//batch_size + 1}. Total processed: {min(i + batch_size, len(documents))}/{len(documents)}"
            )
        logger.info("Storage complete!")

    def similarity_search(self, query: str, k: int = 3):
        """Performs a similarity search and returns documents with scores."""
        return self.vectorstore.similarity_search_with_score(query, k=k)
