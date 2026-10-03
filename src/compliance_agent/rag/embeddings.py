"""
Embedding pipeline for the RAG layer.

Uses a lightweight, local, open-source sentence-transformers model through
LangChain's ``HuggingFaceEmbeddings`` wrapper. The model is downloaded once
from the Hugging Face Hub and cached locally; after that, embedding runs fully
offline with no API keys or external inference services.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings

# ~80 MB, 384-dim, fast on CPU. Good general-purpose semantic similarity model.
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=4)
def get_embedding_model(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    device: str = "cpu",
) -> HuggingFaceEmbeddings:
    """
    Return a cached, local HuggingFace embedding model.

    Embeddings are L2-normalized so that cosine similarity and inner product
    are equivalent, which keeps similarity scores in a predictable range.

    Args:
        model_name: Hugging Face model identifier for a sentence-transformers model.
        device: Torch device to run on ("cpu", "cuda", "mps").

    Returns:
        A LangChain ``HuggingFaceEmbeddings`` instance.
    """
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def embed_documents(texts: List[str], model_name: str = DEFAULT_EMBEDDING_MODEL) -> List[List[float]]:
    """Embed a batch of documents (e.g., compliance rule texts)."""
    return get_embedding_model(model_name).embed_documents(texts)


def embed_query(text: str, model_name: str = DEFAULT_EMBEDDING_MODEL) -> List[float]:
    """Embed a single query (e.g., an extracted R code snippet)."""
    return get_embedding_model(model_name).embed_query(text)
