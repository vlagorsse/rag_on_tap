from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from services.config_service import ConfigService
from services.rag_tool import BeerRAGTool


class TestBeerRAGTool:
    @pytest.fixture
    def mock_config(self):
        config = MagicMock(spec=ConfigService)
        config.google_api_key = "fake_key"
        return config

    @patch("services.rag_tool.VectorStoreService")
    @patch("services.rag_tool.RerankerService")
    def test_run_grouping_logic(
        self, mock_reranker_class, mock_vector_store_class, mock_config
    ):
        """Test that the tool groups multiple chunks from the same recipe."""
        # Setup mocks
        mock_vs = mock_vector_store_class.return_value
        mock_rr = mock_reranker_class.return_value

        # Two chunks from the same beer, one from another
        doc1 = Document(
            page_content="Content 1",
            metadata={"beer_id": "1", "name": "Beer A", "style": "IPA"},
        )
        doc2 = Document(
            page_content="Content 2",
            metadata={"beer_id": "1", "name": "Beer A", "style": "IPA"},
        )
        doc3 = Document(
            page_content="Content 3",
            metadata={"beer_id": "2", "name": "Beer B", "style": "Stout"},
        )

        mock_vs.similarity_search.return_value = [doc1, doc2, doc3]
        mock_rr.rerank.return_value = [(doc1, 0.9), (doc2, 0.8), (doc3, 0.7)]

        tool = BeerRAGTool(
            config=mock_config, model_name="m", collection_name="c", rerank_model="r"
        )
        result = tool._run("test query")

        # Check that Beer A appears once as a header but contains both snippets
        assert "Recipe: Beer A (IPA)" in result
        assert "Recipe: Beer B (Stout)" in result
        assert "Content 1" in result
        assert "Content 2" in result
        assert result.count("Recipe: Beer A") == 1

        # Check URL generation
        assert "https://www.brewersfriend.com/homebrew/recipe/view/1" in result
        assert "https://www.brewersfriend.com/homebrew/recipe/view/2" in result

    def test_get_recipe_url(self, mock_config):
        """Verify URL construction."""
        with patch("services.rag_tool.VectorStoreService"), patch(
            "services.rag_tool.RerankerService"
        ):
            tool = BeerRAGTool(
                config=mock_config,
                model_name="m",
                collection_name="c",
                rerank_model="r",
            )
            assert (
                tool._get_recipe_url("12345")
                == "https://www.brewersfriend.com/homebrew/recipe/view/12345"
            )
            assert tool._get_recipe_url("") == "Unknown URL"
