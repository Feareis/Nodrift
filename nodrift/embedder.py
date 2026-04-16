"""Semantic text embeddings with multiple backend support.

This module provides abstractions for generating sentence embeddings using
different backends (local sentence-transformers or remote OpenAI API).
All embedders implement a consistent interface for use in drift calculations.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class Embedding:
    """Immutable representation of a semantic embedding vector.

    Attributes:
        vector: Normalized embedding vector (typically 384-1536 dimensions).
        model: Identifier of the model that generated this embedding.
        text_length: Length of the original text (for caching/optimization).
    """

    vector: np.ndarray
    model: str
    text_length: int

    @property
    def dimension(self) -> int:
        """Return the dimensionality of the embedding vector."""
        return len(self.vector)

    def __post_init__(self) -> None:
        """Validate vector is 1-dimensional."""
        if self.vector.ndim != 1:
            raise ValueError(
                f"Embedding vector must be 1-dimensional, got shape {self.vector.shape}"
            )


class BaseEmbedder(ABC):
    """Abstract base class for embedding backends.

    Subclasses must implement the embed() method and define MODEL_NAME.
    """

    MODEL_NAME: str

    @abstractmethod
    def embed(self, text: str) -> Embedding:
        """Generate embedding for the given text.

        Args:
            text: Input text to embed.

        Returns:
            Embedding object with vector and metadata.
        """
        pass


class LocalEmbedder(BaseEmbedder):
    """Generate embeddings using sentence-transformers (local/offline).

    Uses the lightweight all-MiniLM-L6-v2 model for 384-dimensional embeddings.
    No external API calls required.
    """

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self) -> None:
        """Initialize the local embedding model.

        Raises:
            ImportError: If sentence-transformers is not installed.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            ) from e

        self._model = SentenceTransformer(self.MODEL_NAME)

    def embed(self, text: str) -> Embedding:
        """Generate embedding for text using local model.

        Args:
            text: Input text to embed.

        Returns:
            Embedding with 384-dimensional vector.
        """
        text_stripped = text.strip()

        if not text_stripped:
            # Return zero vector for empty input
            vector = np.zeros(self._model.get_embedding_dimension(), dtype=np.float32)
        else:
            vector = self._model.encode(text_stripped, convert_to_numpy=True)

        return Embedding(
            vector=vector,
            model=self.MODEL_NAME,
            text_length=len(text),
        )


class OpenAIEmbedder(BaseEmbedder):
    """Generate embeddings using OpenAI's API.

    Uses text-embedding-3-small for 1536-dimensional embeddings.
    Requires OPENAI_API_KEY environment variable or explicit api_key parameter.
    """

    MODEL_NAME = "text-embedding-3-small"

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize OpenAI embedder.

        Args:
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.

        Raises:
            ImportError: If openai package is not installed.
            ValueError: If no API key is provided or found in environment.
        """
        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            ) from e

        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "OpenAI API key required. Provide via api_key= parameter "
                "or set OPENAI_API_KEY environment variable."
            )

        self._client = openai.OpenAI(api_key=self._api_key)

    def embed(self, text: str) -> Embedding:
        """Generate embedding for text using OpenAI API.

        Args:
            text: Input text to embed.

        Returns:
            Embedding with 1536-dimensional vector.

        Raises:
            openai.APIError: If the API request fails.
        """
        text_stripped = text.strip()

        if not text_stripped:
            # Return zero vector for empty input
            vector = np.zeros(1536, dtype=np.float32)
        else:
            response = self._client.embeddings.create(
                input=text_stripped,
                model=self.MODEL_NAME,
            )
            vector = np.array(response.data[0].embedding, dtype=np.float32)

        return Embedding(
            vector=vector,
            model=self.MODEL_NAME,
            text_length=len(text),
        )


class EmbedderFactory:
    """Factory for creating embedder instances.

    Provides a unified interface for instantiating different embedder backends
    with proper error handling and configuration management.
    """

    _backends: dict[str, type[BaseEmbedder]] = {
        "local": LocalEmbedder,
        "openai": OpenAIEmbedder,
    }

    @classmethod
    def create(
        cls,
        mode: Literal["local", "openai"] = "local",
        api_key: str | None = None,
    ) -> BaseEmbedder:
        """Create an embedder instance.

        Args:
            mode: Backend to use ('local' or 'openai').
            api_key: Optional API key for OpenAI backend.

        Returns:
            Configured embedder instance.

        Raises:
            ValueError: If mode is not recognized.
            ImportError: If required dependencies are missing.
        """
        if mode not in cls._backends:
            available = ", ".join(cls._backends.keys())
            raise ValueError(
                f"Unknown embedder mode: '{mode}'. Available: {available}"
            )

        embedder_class = cls._backends[mode]

        if mode == "openai":
            return embedder_class(api_key=api_key)
        return embedder_class()


# Convenience function for backward compatibility
def get_embedder(
    mode: Literal["local", "openai"] = "local",
    api_key: str | None = None,
) -> BaseEmbedder:
    """Create embedder instance (convenience wrapper).

    Deprecated: Use EmbedderFactory.create() directly.
    """
    return EmbedderFactory.create(mode=mode, api_key=api_key)
