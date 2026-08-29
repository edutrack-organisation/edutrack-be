"""Lazy access to local sentence-transformer models."""

from functools import lru_cache

from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def get_question_embedding_model() -> SentenceTransformer:
    """Load the question embedding model only when embeddings are requested."""
    return SentenceTransformer("all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def get_topic_embedding_model() -> SentenceTransformer:
    """Load the larger topic model only for training or topic prediction."""
    return SentenceTransformer("all-mpnet-base-v2")


def encode_question(text: str) -> list[float]:
    return get_question_embedding_model().encode(text).tolist()
