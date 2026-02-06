import logging
from typing import Type

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from services.config_service import ConfigService
from services.reranker_service import RerankerService
from services.vector_store_service import VectorStoreService

logger = logging.getLogger(__name__)


class BeerSearchInput(BaseModel):
    query: str = Field(
        description="The search query for beer recipes, ingredients, or brewing techniques."
    )


class BeerRAGTool(BaseTool):
    name: str = "search_beer_recipes"
    description: str = (
        "Useful for searching specific beer recipes, styles, ingredients, and brewing steps. "
        "Returns the most relevant recipe sections with metadata and source URLs."
    )
    args_schema: Type[BaseModel] = BeerSearchInput

    # Internal services
    _vector_store: VectorStoreService = None
    _reranker: RerankerService = None

    def __init__(
        self,
        config: ConfigService,
        model_name: str,
        collection_name: str,
        rerank_model: str,
    ):
        super().__init__()
        self._vector_store = VectorStoreService(
            config=config, model_name=model_name, collection_name=collection_name
        )
        self._reranker = RerankerService(model_name=rerank_model)

    def _get_recipe_url(self, beer_id: str) -> str:
        """Constructs the original Brewer's Friend URL from the beer ID."""
        if not beer_id:
            return "Unknown URL"
        return f"https://www.brewersfriend.com/homebrew/recipe/view/{beer_id}"

    def _run(self, query: str) -> str:
        """Execute the tool."""
        try:
            logger.info(f"Tool searching for: {query}")
            # 1. Similarity search (candidates)
            initial_results = self._vector_store.similarity_search(query, k=10)

            # 2. Rerank
            results = self._reranker.rerank(query, initial_results, top_k=3)

            if not results:
                return "No relevant beer recipes found for this query."

            # 3. Group results by Recipe to avoid redundancy
            # Key: beer_id, Value: {name, style, url, contents[]}
            grouped: dict[str, dict] = {}

            for doc, score in results:
                meta = doc.metadata
                bid = meta.get("beer_id")
                if not bid:
                    continue

                if bid not in grouped:
                    grouped[bid] = {
                        "name": meta.get("name", "Unknown Recipe"),
                        "style": meta.get("style", "Unknown Style"),
                        "url": self._get_recipe_url(bid),
                        "contents": [],
                    }

                # Use the page_content (which already has contextual headers from ChunkingService)
                grouped[bid]["contents"].append(doc.page_content)

            # 4. Format output
            formatted_outputs = []
            for bid, data in grouped.items():
                # Join multiple snippets from the same recipe with double newlines
                combined_content = "\n\n".join(data["contents"])
                output = (
                    f"--- Recipe: {data['name']} ({data['style']}) ---\n"
                    f"URL: {data['url']}\n"
                    f"Details:\n{combined_content}\n"
                )
                formatted_outputs.append(output)

            return "\n".join(formatted_outputs)

        except Exception as e:
            logger.error(f"Error in BeerRAGTool: {e}")
            return f"An error occurred while searching for recipes: {str(e)}"

    async def _arun(self, query: str) -> str:
        """Async version of the tool."""
        return self._run(query)
