import logging

import torch
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class RerankerService:
    def __init__(self, model_name: str = "Qwen/Qwen3-Reranker-0.6B"):
        logger.info(f"Loading reranker model ({model_name})...")

        # Use sentence-transformers directly for better control
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(model_name, trust_remote_code=True, device=device)

        # Explicitly set pad_token to eos_token to fix the batching error
        # Qwen3 models often don't have a default pad_token in their config
        self.model.tokenizer.pad_token = self.model.tokenizer.eos_token
        self.model.model.config.pad_token_id = self.model.tokenizer.eos_token_id

        logger.info(
            f"Reranker initialized. Pad token: {self.model.tokenizer.pad_token} (ID: {self.model.model.config.pad_token_id})"
        )

    def rerank(self, query: str, results: list, top_k: int = 3):
        """
        Reranks a list of (Document, initial_score) results using a cross-encoder.
        """
        if not results:
            return []

        # results is a list of (Document, score) from similarity_search_with_score
        docs = [res[0] for res in results]

        # Prepare pairs for the cross-encoder: (query, passage)
        pairs = [[query, doc.page_content] for doc in docs]

        # Get scores from the cross-encoder
        scores = self.model.predict(pairs)

        # Combine documents with reranked scores and sort
        doc_scores = list(zip(docs, scores))
        reranked = sorted(doc_scores, key=lambda x: x[1], reverse=True)

        return reranked[:top_k]
