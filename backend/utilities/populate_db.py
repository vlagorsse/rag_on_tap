import argparse
import logging
from typing import List

from langchain_core.documents import Document

from services.chunking_service import ChunkingService
from services.config_service import ConfigService
from services.data_service import DataService
from services.file_dump_service import FileDumpService
from services.storage_service import StorageService
from services.vector_store_service import VectorStoreService

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

COLLECTION_NAME_DEFAULT = "beer_recipes"


def load_documents_from_csv(csv_path: str, limit: int | None = None) -> List[Document]:
    """Loads enriched beer recipes from a CSV file and converts them to LangChain Documents."""
    logger.info(f"Loading data from {csv_path}...")
    data_service = DataService(csv_path)
    df = data_service.load()

    if df.empty:
        logger.warning(f"No data loaded from {csv_path}.")
        return []

    if limit is not None:
        logger.info(f"Limiting to first {limit} rows.")
        df = df.head(limit)

    df = df.fillna("")

    if "enriched_story" not in df.columns:
        raise ValueError(f"Column 'enriched_story' not found in {csv_path}")

    documents = []
    for _, row in df.iterrows():
        content = row["enriched_story"]
        if not content:
            logger.warning(
                f"enriched_story is empty for beer {row['BeerID']} ({row['Name']}). Skipping."
            )
            continue

        metadata = {
            "beer_id": str(row["BeerID"]),
            "name": row["Name"],
            "style": row["Style"],
            "abv": row["ABV"],
            "ibu": row["IBU"],
        }

        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)

    logger.info(f"Loaded {len(documents)} documents.")
    return documents


def populate_db(
    csv_path: str,
    limit: int | None,
    storage_service: StorageService,
    chunk_size: int = 500,
    chunk_overlap: int = 0,
):
    """Orchestrates the loading, splitting, and storing process using dependency injection."""

    documents = load_documents_from_csv(csv_path, limit)
    if not documents:
        logger.error("No documents to process.")
        return

    # Chunking
    chunking_service = ChunkingService(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    split_docs = chunking_service.split_documents(documents)

    # Storage via injected service
    storage_service.add_documents(split_docs)


def run_population(args: argparse.Namespace):
    """Handles service selection and dependency injection based on arguments."""
    if args.dry_run:
        logger.info(f"Dry run enabled. Output will be saved to {args.dry_run}")
        storage_service = FileDumpService(args.dry_run)
    else:
        config = ConfigService()
        storage_service = VectorStoreService(
            config=config, model_name=args.model, collection_name=args.collection
        )

    populate_db(
        csv_path=args.csv_path,
        limit=args.limit,
        storage_service=storage_service,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )


def main():
    """Handles CLI argument parsing."""
    parser = argparse.ArgumentParser(description="Populate PGVector database from CSV")
    parser.add_argument("csv_path", help="Path to the enriched recipes CSV")
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help="Limit the number of rows to process",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="Qwen/Qwen3-Embedding-0.6B",
        help="Embedding model name to use (default: Qwen/Qwen3-Embedding-0.6B)",
    )
    parser.add_argument(
        "--collection",
        "-c",
        default=COLLECTION_NAME_DEFAULT,
        help=f"Collection name (default: {COLLECTION_NAME_DEFAULT})",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=500,
        help="Chunk size for splitting documents (default: 500)",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=0,
        help="Chunk overlap for splitting documents (default: 0)",
    )
    parser.add_argument(
        "--dry-run",
        type=str,
        metavar="FILE_PATH",
        help="Do not populate DB; instead, dump chunks to this JSONL file path",
    )
    args = parser.parse_args()

    run_population(args)


if __name__ == "__main__":
    main()
