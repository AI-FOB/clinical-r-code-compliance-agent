"""
Retrieval evaluation tests for the v0.2 RAG layer.

Note: the first run downloads the local embedding model (~80 MB) from the
Hugging Face Hub; subsequent runs use the local cache.
"""

import os

import pytest

from src.compliance_agent.rag.retriever import RuleRetriever
from src.compliance_agent.rag.vector_store import RuleVectorStore

RULES_DIR = os.path.join(os.path.dirname(__file__), "..", "knowledge", "rules")


@pytest.fixture(scope="module")
def retriever():
    store = RuleVectorStore(rules_dir=RULES_DIR, collection_name="test_compliance_rules")
    store.build_index()
    return RuleRetriever(vector_store=store, top_k=3)


def test_all_rules_indexed(retriever):
    n_rule_files = len([f for f in os.listdir(RULES_DIR) if f.endswith(".json")])
    assert retriever.vector_store.count() == n_rule_files


def test_hardcoded_treatment_is_top_match(retriever):
    results = retriever.retrieve("TRT == 'ACTIVE'")

    assert len(results) == 3
    top = results[0]
    assert top.source_file == "hardcoded_treatment.json"
    assert top.rule_id == "COMP-001"
    # Results are ordered by descending similarity.
    assert all(results[i].score >= results[i + 1].score for i in range(len(results) - 1))


def test_empty_snippet_rejected(retriever):
    with pytest.raises(ValueError):
        retriever.retrieve("   ")
