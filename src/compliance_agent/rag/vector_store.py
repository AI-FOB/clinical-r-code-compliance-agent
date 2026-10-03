"""
Local ChromaDB vector store for synthetic compliance rules.

Loads every JSON rule in ``knowledge/rules/``, converts each rule into a
natural-language document, embeds it with the local embedding model, and
indexes it in a ChromaDB collection (cosine similarity).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings
from langchain_core.embeddings import Embeddings

from .embeddings import get_embedding_model

DEFAULT_RULES_DIR = Path(__file__).resolve().parents[3] / "knowledge" / "rules"
DEFAULT_COLLECTION_NAME = "compliance_rules"


def load_rules(rules_dir: Path | str = DEFAULT_RULES_DIR) -> List[Dict[str, Any]]:
    """
    Load all JSON rule files from ``rules_dir``.

    Each returned rule dict is augmented with a ``source_file`` key holding
    the rule's file name (e.g., ``hardcoded_treatment.json``).
    """
    rules_path = Path(rules_dir)
    if not rules_path.is_dir():
        raise FileNotFoundError(f"Rules directory not found: {rules_path}")

    rules: List[Dict[str, Any]] = []
    for file_path in sorted(rules_path.glob("*.json")):
        with open(file_path, "r", encoding="utf-8") as f:
            rule = json.load(f)
        rule["source_file"] = file_path.name
        rules.append(rule)
    return rules


def rule_to_document(rule: Dict[str, Any]) -> str:
    """
    Render a rule as text for embedding.

    Includes the name, description, recommendation, and detection patterns so
    that both natural-language and code-like queries can match the rule.
    """
    parts = [
        f"Rule {rule.get('rule_id', '')}: {rule.get('rule_name', '')}",
        f"Description: {rule.get('description', '')}",
        f"Recommendation: {rule.get('recommendation', '')}",
    ]
    patterns = rule.get("patterns") or []
    if patterns:
        parts.append("Detection patterns: " + " ; ".join(patterns))
    keywords = rule.get("required_keywords") or []
    if keywords:
        parts.append("Required keywords: " + ", ".join(keywords))
    return "\n".join(parts)


class RuleVectorStore:
    """ChromaDB-backed index of compliance rules."""

    def __init__(
        self,
        rules_dir: Path | str = DEFAULT_RULES_DIR,
        persist_directory: Optional[Path | str] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model: Optional[Embeddings] = None,
    ) -> None:
        """
        Args:
            rules_dir: Directory containing JSON rule files.
            persist_directory: If given, store the index on disk; otherwise in memory.
            collection_name: ChromaDB collection name.
            embedding_model: LangChain embeddings; defaults to the local HF model.
        """
        self.rules_dir = Path(rules_dir)
        self.collection_name = collection_name
        self.embedding_model = embedding_model or get_embedding_model()

        settings = Settings(anonymized_telemetry=False, allow_reset=True)
        if persist_directory is not None:
            self.client = chromadb.PersistentClient(path=str(persist_directory), settings=settings)
        else:
            self.client = chromadb.EphemeralClient(settings=settings)

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def build_index(self, rebuild: bool = True) -> int:
        """
        Load, embed, and index all rules.

        Args:
            rebuild: Drop and recreate the collection first so removed rules
                do not linger in the index.

        Returns:
            Number of rules indexed.
        """
        if rebuild:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

        rules = load_rules(self.rules_dir)
        if not rules:
            return 0

        documents = [rule_to_document(r) for r in rules]
        embeddings = self.embedding_model.embed_documents(documents)
        ids = [r.get("rule_id") or r["source_file"] for r in rules]
        metadatas = [
            {
                "rule_id": r.get("rule_id", ""),
                "rule_name": r.get("rule_name", ""),
                "severity": r.get("severity", ""),
                "source_file": r["source_file"],
                "rule_json": json.dumps(r),
            }
            for r in rules
        ]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        return len(rules)

    def count(self) -> int:
        """Number of indexed rules."""
        return self.collection.count()

    def query(self, query_embedding: List[float], k: int = 3) -> Dict[str, Any]:
        """Raw nearest-neighbour query against the collection."""
        n = max(1, min(k, self.count()))
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
