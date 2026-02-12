import logging

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class ChunkingService:
    """Handles splitting structured documents into smaller chunks with contextual headers."""

    def __init__(self):
        """Initializes the service. Logic is strictly based on document structure (\n\n)."""
        pass

    def _split_section_and_content(self, text: str) -> tuple[str, str]:
        """Splits the text into (section_name, actual_content)."""
        if ":" in text:
            section, content = text.split(":", 1)
            return section.strip(), content.strip()
        return "General", text.strip()

    def _create_contextual_content(self, section_text: str, metadata: dict) -> str:
        """Prepends a contextual header and cleans the section text."""
        name = metadata.get("name", "Unknown Recipe")
        style = metadata.get("style", "Unknown Style")
        section, clean_content = self._split_section_and_content(section_text)

        return f"Recipe: {name} | Style: {style} | Section: {section} | Text: {clean_content}"

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Splits a list of documents strictly at every double newline and adds context headers."""
        if not documents:
            return []

        logger.info(f"Splitting {len(documents)} documents into contextual chunks...")

        split_docs = []
        for doc in documents:
            sections = doc.page_content.split("\n\n")

            current_offset = 0
            for section in sections:
                if not section.strip():
                    current_offset += len(section) + 2
                    continue

                # Prepend the contextual header and clean content
                contextual_content = self._create_contextual_content(
                    section, doc.metadata
                )

                new_doc = Document(
                    page_content=contextual_content, metadata=doc.metadata.copy()
                )
                # start_index refers to the location in the ORIGINAL raw story
                new_doc.metadata["start_index"] = current_offset
                # Store the raw section text in metadata for clean UI display if needed
                new_doc.metadata["raw_content"] = section.strip()

                split_docs.append(new_doc)
                current_offset += len(section) + 2

        logger.info(f"Created {len(split_docs)} contextual chunks.")
        return split_docs
