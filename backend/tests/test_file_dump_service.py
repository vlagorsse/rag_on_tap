import json
import os

import pytest
from langchain_core.documents import Document

from services.file_dump_service import FileDumpService


class TestFileDumpService:
    @pytest.fixture
    def output_path(self, tmp_path):
        """Fixture to provide a temporary file path."""
        return str(tmp_path / "test_dump.jsonl")

    def test_add_documents(self, output_path):
        """Test that add_documents correctly writes JSONL data."""
        service = FileDumpService(output_path)

        docs = [
            Document(page_content="Beer 1 info", metadata={"id": 1}),
            Document(page_content="Beer 2 info", metadata={"id": 2}),
        ]

        service.add_documents(docs)

        # Verify file exists and content is correct
        assert os.path.exists(output_path)

        with open(output_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) == 2

        data1 = json.loads(lines[0])
        assert data1["page_content"] == "Beer 1 info"
        assert data1["metadata"]["id"] == 1

        data2 = json.loads(lines[1])
        assert data2["page_content"] == "Beer 2 info"
        assert data2["metadata"]["id"] == 2

    def test_add_empty_list(self, output_path):
        """Test that adding an empty list creates an empty file (or does nothing)."""
        service = FileDumpService(output_path)
        service.add_documents([])

        # In the current implementation, it creates the file even if empty
        assert os.path.exists(output_path)
        with open(output_path, "r") as f:
            assert f.read() == ""
