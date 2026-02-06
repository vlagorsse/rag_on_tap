import argparse
import logging

from services.config_service import ConfigService
from services.reranker_service import RerankerService
from services.vector_store_service import VectorStoreService

logger = logging.getLogger(__name__)


def query_loop(collection_name: str, model_name: str, rerank_model: str):
    config = ConfigService()

    vector_store_service = VectorStoreService(
        config=config, model_name=model_name, collection_name=collection_name
    )

    reranker = RerankerService(model_name=rerank_model)

    print("\n--- Beer RAG Query Shell ---")
    print("Type your query and press Enter. Type 'exit' or 'quit' to leave.\n")

    while True:
        try:
            query = input("Query> ")
            if query.lower() in ("exit", "quit"):
                break
            if not query.strip():
                continue

            # Fetch candidates for reranking
            logger.info(f"Searching for candidates for query: {query}")
            initial_results = vector_store_service.similarity_search(query, k=10)

            print(f"Reranking candidates...")
            results = reranker.rerank(query, initial_results, top_k=3)
            # results = initial_results

            print(f"\nTop {len(results)} relevant recipes (after reranking):\n")
            for i, (doc, score) in enumerate(results, 1):
                meta = doc.metadata
                print(f"--- Result {i} (Rerank Score: {score:.4f}) ---")
                print(f"Name: {meta.get('name', 'Unknown')}")
                print(f"Style: {meta.get('style', 'Unknown')}")
                print(f"ABV: {meta.get('abv', 'N/A')}% | IBU: {meta.get('ibu', 'N/A')}")
                print(f"Excerpt: {doc.page_content[:200]}...")
                print()

        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
        except Exception as e:
            logger.exception("An error occurred during query processing")
            print(f"Error: {e}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # Filter out noisy library logs
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Query the Beer RAG database")
    parser.add_argument(
        "--collection",
        "-c",
        default="beer_recipes",
        help="Collection name (default: beer_recipes)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="Qwen/Qwen3-Embedding-0.6B",
        help="Embedding model to use (default: Qwen/Qwen3-Embedding-0.6B)",
    )
    parser.add_argument(
        "--rerank-model",
        "-r",
        default="Qwen/Qwen3-Reranker-0.6B",
        help="Reranker model to use (default: Qwen/Qwen3-Reranker-0.6B)",
    )
    args = parser.parse_args()

    query_loop(args.collection, args.model, args.rerank_model)


if __name__ == "__main__":
    main()
