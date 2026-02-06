from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from services.reranker_service import RerankerService


@pytest.fixture
def mock_cross_encoder():
    with patch("services.reranker_service.CrossEncoder") as mock:
        yield mock


def test_reranker_initialization(mock_cross_encoder):
    # Setup mock instance
    mock_instance = MagicMock()
    mock_instance.tokenizer.pad_token = None
    mock_instance.tokenizer.eos_token = "<eos>"
    mock_instance.tokenizer.eos_token_id = 123
    mock_instance.model.config.eos_token_id = 123

    mock_cross_encoder.return_value = mock_instance

    RerankerService(model_name="test-model")

    # Check that pad_token and pad_token_id were set
    assert mock_instance.tokenizer.pad_token == "<eos>"
    assert mock_instance.model.config.pad_token_id == 123


def test_rerank_logic(mock_cross_encoder):
    # Setup mock scores
    mock_instance = MagicMock()
    mock_instance.predict.return_value = [0.1, 0.9, 0.5]
    mock_cross_encoder.return_value = mock_instance

    service = RerankerService()

    query = "IPA recipe"
    docs = [
        (Document(page_content="doc1"), 0.9),
        (Document(page_content="doc2"), 0.1),
        (Document(page_content="doc3"), 0.5),
    ]

    results = service.rerank(query, docs, top_k=2)

    assert len(results) == 2
    assert results[0][0].page_content == "doc2"  # Highest mock score 0.9
    assert results[1][0].page_content == "doc3"  # Second highest 0.5


def test_rerank_empty_list(mock_cross_encoder):
    mock_cross_encoder.return_value = MagicMock()
    service = RerankerService()
    results = service.rerank("query", [], top_k=3)
    assert results == []
