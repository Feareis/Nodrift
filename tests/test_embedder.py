"""Tests pour nodrift.embedder."""

import numpy as np
import pytest

from nodrift.embedder import LocalEmbedder, get_embedder


class TestLocalEmbedder:
    """Tests du LocalEmbedder (sentence-transformers)."""

    @pytest.fixture
    def embedder(self):
        return LocalEmbedder()

    def test_embed_returns_embedding(self, embedder):
        result = embedder.embed("Hello world")
        assert result.vector is not None
        assert len(result.vector) > 0
        assert result.model == "sentence-transformers/all-MiniLM-L6-v2"

    def test_embed_consistent_dimension(self, embedder):
        """Les embeddings ont tous la même dimension."""
        emb1 = embedder.embed("This is a test.")
        emb2 = embedder.embed("Another test sentence.")
        assert emb1.dimension == emb2.dimension

    def test_embed_empty_text_returns_zero_vector(self, embedder):
        result = embedder.embed("")
        assert np.allclose(result.vector, 0)

    def test_embed_whitespace_only_returns_zero_vector(self, embedder):
        result = embedder.embed("   \n\n  ")
        assert np.allclose(result.vector, 0)

    def test_embed_stores_text_length(self, embedder):
        text = "Hello world"
        result = embedder.embed(text)
        assert result.text_length == len(text)

    def test_embed_similar_texts_similar_vectors(self, embedder):
        """Textes similaires → vecteurs similaires."""
        emb1 = embedder.embed("The cat sat on the mat")
        emb2 = embedder.embed("A cat was sitting on a mat")

        # Cosine similarity should be high (> 0.7)
        sim = np.dot(emb1.vector, emb2.vector) / (
            np.linalg.norm(emb1.vector) * np.linalg.norm(emb2.vector)
        )
        assert sim > 0.7

    def test_embed_different_texts_different_vectors(self, embedder):
        """Textes très différents → vecteurs différents."""
        emb1 = embedder.embed("The quick brown fox")
        emb2 = embedder.embed("Quantum computing algorithms")

        sim = np.dot(emb1.vector, emb2.vector) / (
            np.linalg.norm(emb1.vector) * np.linalg.norm(emb2.vector)
        )
        assert sim < 0.7


class TestFactory:
    """Tests pour la fonction get_embedder()."""

    def test_get_embedder_local_returns_local_embedder(self):
        embedder = get_embedder(mode="local")
        assert isinstance(embedder, LocalEmbedder)

    def test_get_embedder_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown embedder mode"):
            get_embedder(mode="invalid")

    def test_get_embedder_openai_requires_api_key(self):
        """OpenAI mode sans clé API devrait lever."""
        import os

        # Supprimer la clé si elle existe
        api_key_backup = os.environ.pop("OPENAI_API_KEY", None)
        try:
            # Lève ImportError si openai n'est pas installé,
            # ou ValueError si la clé API manque
            with pytest.raises((ImportError, ValueError)):
                get_embedder(mode="openai")
        finally:
            if api_key_backup:
                os.environ["OPENAI_API_KEY"] = api_key_backup
