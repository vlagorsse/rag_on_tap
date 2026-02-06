from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from services.config_service import ConfigService
from services.vector_store_service import VectorStoreService


class TestVectorStoreService:
    @pytest.fixture
    def mock_config(self):
        config = MagicMock(spec=ConfigService)
        config.connection_string = "postgresql+psycopg://user:pass@host:5432/db"
        return config

    @patch("services.vector_store_service.HuggingFaceEmbeddings")
    @patch("services.vector_store_service.PGVector")
    def test_initialization(self, mock_pgvector, mock_embeddings, mock_config):
        """Test that the service initializes embeddings and PGVector correctly."""
        service = VectorStoreService(
            config=mock_config,
            model_name="test-model",
            collection_name="test-collection",
        )

        # Verify embeddings init
        mock_embeddings.assert_called_once_with(
            model_name="test-model", model_kwargs={}
        )

        # Verify PGVector init
        mock_pgvector.assert_called_once_with(
            embeddings=service.embeddings,
            collection_name="test-collection",
            connection=mock_config.connection_string,
            use_jsonb=True,
        )

    @patch("services.vector_store_service.HuggingFaceEmbeddings")
    @patch("services.vector_store_service.PGVector")
    def test_add_documents(self, mock_pgvector, mock_embeddings, mock_config):
        """Test that add_documents calls the underlying vectorstore."""
        service = VectorStoreService(config=mock_config)
        mock_vs = mock_pgvector.return_value

        docs = [Document(page_content="test content")]
        service.add_documents(docs, batch_size=1)

        mock_vs.add_documents.assert_called_once_with(docs)

    @patch("services.vector_store_service.HuggingFaceEmbeddings")
    @patch("services.vector_store_service.PGVector")
    def test_similarity_search(self, mock_pgvector, mock_embeddings, mock_config):
        """Test that similarity_search calls the underlying vectorstore."""
        service = VectorStoreService(config=mock_config)
        mock_vs = mock_pgvector.return_value
        mock_vs.similarity_search_with_score.return_value = []

        query = "beer"
        service.similarity_search(query, k=5)

        mock_vs.similarity_search_with_score.assert_called_once_with(query, k=5)
