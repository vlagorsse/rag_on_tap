import logging
from enum import Enum
from typing import Type

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from services.config_service import ConfigService
from services.reranker_service import RerankerService
from services.vector_store_service import VectorStoreService

logger = logging.getLogger(__name__)


class StyleEnum(str, Enum):
    American_Brown_Ale = "American Brown Ale"
    American_IPA = "American IPA"
    American_Light_Lager = "American Light Lager"
    American_Pale_Ale = "American Pale Ale"
    American_Wheat_or_Rye_Beer = "American Wheat or Rye Beer"
    Belgian_Blond_Ale = "Belgian Blond Ale"
    Belgian_Dark_Strong_Ale = "Belgian Dark Strong Ale"
    Belgian_Golden_Strong_Ale = "Belgian Golden Strong Ale"
    Belgian_Tripel = "Belgian Tripel"
    Blonde_Ale = "Blonde Ale"
    Bohemian_Pilsener = "Bohemian Pilsener"
    Cream_Ale = "Cream Ale"
    Czech_Premium_Pale_Lager = "Czech Premium Pale Lager"
    Double_IPA = "Double IPA"
    Dry_Stout = "Dry Stout"
    English_IPA = "English IPA"
    Extra_Special_Strong_Bitter = "Extra Special/Strong Bitter (ESB)"
    Fruit_Beer = "Fruit Beer"
    German_Pilsner = "German Pilsner (Pils)"
    Holiday_Winter_Special_Spiced_Beer = "Holiday/Winter Special Spiced Beer"
    Imperial_IPA = "Imperial IPA"
    International_Pale_Lager = "International Pale Lager"
    Irish_Red_Ale = "Irish Red Ale"
    Kölsch = "Kölsch"
    Light_American_Lager = "Light American Lager"
    Mixed_Fermentation_Sour_Beer = "Mixed-Fermentation Sour Beer"
    Northern_English_Brown = "Northern English Brown"
    Oatmeal_Stout = "Oatmeal Stout"
    Oktoberfest_Märzen = "Oktoberfest/Märzen"
    Premium_American_Lager = "Premium American Lager"
    Robust_Porter = "Robust Porter"
    Saison = "Saison"
    Specialty_IPA_New_England_IPA = "Specialty IPA: New England IPA"
    Sweet_Stout = "Sweet Stout"
    Weissbier = "Weissbier"
    Weizen_Weissbier = "Weizen/Weissbier"
    Winter_Seasonal_Beer = "Winter Seasonal Beer"
    Witbier = "Witbier"


class BeerSearchInput(BaseModel):
    query: str = Field(
        description="The search query for beer recipes, ingredients, or brewing techniques."
    )
    styles: list[StyleEnum] | None = Field(
        default=None, description="Filter query by one or more beer styles"
    )
    abv_lte: float | None = Field(
        default=None,
        description="Filter search query for beer with less than or equal by alcohol by volume.",
    )
    abv_gt: float | None = Field(
        default=None,
        description="Filter search query for beer with greater by alcohol by volume.",
    )
    ibu_lte: float | None = Field(
        default=None,
        description="Filter search query for beer with bitterness less than or equal by IBU.",
    )
    ibu_gt: float | None = Field(
        default=None,
        description="Filter search query for beer with bitterness greater than alcohol by volume by IBU.",
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

    def _run(
        self,
        query: str,
        styles: list[StyleEnum] | None = None,
        abv_lte: float | None = None,
        abv_gt: float | None = None,
        ibu_lte: float | None = None,
        ibu_gt: float | None = None,
    ) -> str:
        """Execute the tool."""
        try:
            logger.info(
                f"Tool searching for: query {query}, styles {styles}, abv_lte {abv_lte}, abv_gt {abv_gt}, ibu_lte {ibu_lte}, ibu_gt {ibu_gt}"
            )

            # Construct metadata filtering
            filters = []
            if styles:
                if len(styles) == 1:
                    filters.append({"style": styles[0]})
                else:
                    filters.append({"style": {"$in": styles}})
            if abv_lte is not None:
                filters.append({"abv": {"$lte": abv_lte}})
            if abv_gt is not None:
                filters.append({"abv": {"$gt": abv_gt}})
            if ibu_lte is not None:
                filters.append({"ibu": {"$lte": ibu_lte}})
            if ibu_gt is not None:
                filters.append({"ibu": {"$gt": ibu_gt}})

            filter = {}
            if len(filters) == 1:
                filter = filters[0]
            elif len(filters) > 1:
                filter = {"$and": filters}
            else:
                filter = None

            # 1. Similarity search (candidates)
            initial_results = self._vector_store.similarity_search(
                query, k=10, filter=filter
            )

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

    async def _arun(
        self,
        query: str,
        styles: list[StyleEnum] | None = None,
        abv_lte: float | None = None,
        abv_gt: float | None = None,
        ibu_lte: float | None = None,
        ibu_gt: float | None = None,
    ) -> str:
        """Async version of the tool."""
        return self._run(
            query,
            styles=styles,
            abv_lte=abv_lte,
            abv_gt=abv_gt,
            ibu_lte=ibu_lte,
            ibu_gt=ibu_gt,
        )
