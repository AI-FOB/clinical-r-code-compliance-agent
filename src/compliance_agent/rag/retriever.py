"""
Retriever: maps an R code snippet to the most relevant compliance rules.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .vector_store import RuleVectorStore


class RetrievedRule(BaseModel):
    """A compliance rule returned by similarity search."""

    rule_id: str
    rule_name: str
    severity: str
    source_file: str = Field(..., description="Rule file name, e.g. hardcoded_treatment.json")
    score: float = Field(..., description="Cosine similarity (higher is more relevant)")
    rule: Dict[str, Any] = Field(..., description="Full rule definition")


class RuleRetriever:
    """Embeds an R code snippet and returns the top-k most similar rules."""

    def __init__(self, vector_store: Optional[RuleVectorStore] = None, top_k: int = 3) -> None:
        self.vector_store = vector_store or RuleVectorStore()
        if self.vector_store.count() == 0:
            self.vector_store.build_index()
        self.top_k = top_k

    def retrieve(self, code_snippet: str, k: Optional[int] = None) -> List[RetrievedRule]:
        """
        Args:
            code_snippet: Extracted R code (one or more lines).
            k: Number of rules to return (defaults to ``top_k``).

        Returns:
            Rules ordered from most to least relevant.
        """
        if not code_snippet or not code_snippet.strip():
            raise ValueError("code_snippet must be a non-empty string")

        query_embedding = self.vector_store.embedding_model.embed_query(code_snippet)
        result = self.vector_store.query(query_embedding, k=k or self.top_k)

        retrieved: List[RetrievedRule] = []
        for meta, distance in zip(result["metadatas"][0], result["distances"][0]):
            retrieved.append(
                RetrievedRule(
                    rule_id=meta["rule_id"],
                    rule_name=meta["rule_name"],
                    severity=meta["severity"],
                    source_file=meta["source_file"],
                    score=1.0 - float(distance),  # cosine distance -> similarity
                    rule=json.loads(meta["rule_json"]),
                )
            )
        return retrieved
