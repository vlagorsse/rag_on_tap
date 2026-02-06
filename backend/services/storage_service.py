from abc import ABC, abstractmethod

from langchain_core.documents import Document


class StorageService(ABC):
    """Abstract base class for storage services."""

    @abstractmethod
    def add_documents(self, documents: list[Document], batch_size: int = 500):
        """Adds documents to the storage."""
        pass
