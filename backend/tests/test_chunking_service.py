import pytest
from langchain_core.documents import Document

from services.chunking_service import ChunkingService


class TestChunkingService:
    def test_contextual_header_generation_clean_text(self):
        """Test that chunks receive the correct header and the Text: portion is cleaned."""
        service = ChunkingService()

        content = "Appearance Color: Pale straw.\n\nAroma Hop: Citrusy."
        metadata = {"name": "Test Ale", "style": "IPA"}
        doc = Document(page_content=content, metadata=metadata)

        chunks = service.split_documents([doc])

        assert len(chunks) == 2

        # Check first chunk: 'Appearance Color' should be in Section, but NOT in Text
        assert "Section: Appearance Color" in chunks[0].page_content
        assert "Text: Pale straw." in chunks[0].page_content
        assert "Text: Appearance Color:" not in chunks[0].page_content

        # Check second chunk
        assert "Section: Aroma Hop" in chunks[1].page_content
        assert "Text: Citrusy." in chunks[1].page_content

    def test_metadata_preservation(self):
        """Verify that original metadata and start_index are preserved."""
        service = ChunkingService()
        content = "Section 1: Details\n\nSection 2: More"
        doc = Document(
            page_content=content, metadata={"beer_id": "123", "name": "N", "style": "S"}
        )

        chunks = service.split_documents([doc])

        for chunk in chunks:
            assert chunk.metadata["beer_id"] == "123"
            assert "raw_content" in chunk.metadata
            # Verify raw_content still has the category for UI display
            assert ":" in chunk.metadata["raw_content"]

    def test_empty_list(self):
        """Test that ChunkingService returns an empty list for empty input."""
        service = ChunkingService()
        assert service.split_documents([]) == []
