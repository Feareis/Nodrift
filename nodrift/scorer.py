"""Semantic drift scoring and comparison between prompt versions.

This module compares two parsed prompts using semantic embeddings and
produces detailed drift reports with severity classifications.

Drift Thresholds:
    - 0.00-0.15: OK (cosmetic changes)
    - 0.15-0.40: Warning (notable changes, requires review)
    - 0.40+: Breaking (behavioral change, should block deployment)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.linalg import norm

from nodrift.embedder import BaseEmbedder, EmbedderFactory
from nodrift.parser import ParsedPrompt, PromptSection


class DriftSeverityError(ValueError):
    """Raised when drift calculations produce invalid values."""

    pass


@dataclass(frozen=True)
class SectionDrift:
    """Immutable drift metrics for a single prompt section.

    Attributes:
        name: Section identifier.
        drift_score: Normalized drift (0.0-1.0).
        similarity: Cosine similarity of embeddings (-1.0 to 1.0).
        severity: Classification based on drift_score.
    """

    name: str
    drift_score: float
    similarity: float
    severity: Literal["ok", "warning", "breaking"] = field(init=False)

    def __post_init__(self) -> None:
        """Calculate severity level from drift score."""
        # Clamp to valid range (handles floating-point precision issues)
        clamped_drift = max(0.0, min(1.0, self.drift_score))

        if clamped_drift < 0.15:
            object.__setattr__(self, "severity", "ok")
        elif clamped_drift < 0.40:
            object.__setattr__(self, "severity", "warning")
        else:
            object.__setattr__(self, "severity", "breaking")


@dataclass(frozen=True)
class DriftReport:
    """Immutable comprehensive drift analysis between two prompts.

    Attributes:
        sections: Per-section drift metrics.
        overall_drift: Mean drift across all sections (0.0-1.0).
        overall_severity: Classification of overall drift.
    """

    sections: dict[str, SectionDrift]
    overall_drift: float
    overall_severity: Literal["ok", "warning", "breaking"] = field(init=False)

    def __post_init__(self) -> None:
        """Calculate overall severity from drift score."""
        # Clamp to valid range
        clamped_drift = max(0.0, min(1.0, self.overall_drift))

        if clamped_drift < 0.15:
            object.__setattr__(self, "overall_severity", "ok")
        elif clamped_drift < 0.40:
            object.__setattr__(self, "overall_severity", "warning")
        else:
            object.__setattr__(self, "overall_severity", "breaking")


# ─────────────────────────────────────────────────────────────────
# Similarity Computation
# ─────────────────────────────────────────────────────────────────


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        vec_a: First vector.
        vec_b: Second vector.

    Returns:
        Similarity score in range [-1.0, 1.0].
    """
    norm_a = norm(vec_a)
    norm_b = norm(vec_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def similarity_to_drift(similarity: float) -> float:
    """Convert cosine similarity to drift score.

    Maps the similarity range [-1.0, 1.0] to drift range [0.0, 1.0]:
        - Similarity 1.0 (identical) → drift 0.0
        - Similarity 0.0 (orthogonal) → drift 0.5
        - Similarity -1.0 (opposite) → drift 1.0

    Args:
        similarity: Cosine similarity score.

    Returns:
        Normalized drift score in [0.0, 1.0].
    """
    return (1.0 - similarity) / 2.0


# ─────────────────────────────────────────────────────────────────
# Section Scoring
# ─────────────────────────────────────────────────────────────────


def score_section(
    old: PromptSection,
    new: PromptSection,
    embedder: BaseEmbedder,
) -> SectionDrift:
    """Calculate drift between two sections.

    Args:
        old: Original section.
        new: Modified section.
        embedder: Embedding generator.

    Returns:
        SectionDrift with computed metrics.
    """
    old_text = old.content.strip()
    new_text = new.content.strip()

    # Handle edge cases
    if not old_text and not new_text:
        return SectionDrift(name=old.name, drift_score=0.0, similarity=1.0)

    if not old_text or not new_text:
        return SectionDrift(name=old.name, drift_score=1.0, similarity=-1.0)

    # Compute embeddings and similarity
    old_embedding = embedder.embed(old_text)
    new_embedding = embedder.embed(new_text)

    similarity = cosine_similarity(old_embedding.vector, new_embedding.vector)
    drift = similarity_to_drift(similarity)

    return SectionDrift(name=old.name, drift_score=drift, similarity=similarity)


# ─────────────────────────────────────────────────────────────────
# Main Diff Function
# ─────────────────────────────────────────────────────────────────


def diff(
    old_prompt: ParsedPrompt,
    new_prompt: ParsedPrompt,
    embedder_mode: Literal["local", "openai"] = "local",
    api_key: str | None = None,
) -> DriftReport:
    """Compare two prompts and produce detailed drift report.

    Analyzes all sections present in either prompt version and computes
    semantic drift metrics for each. Sections present in only one version
    are treated as additions/removals with maximum drift.

    Args:
        old_prompt: Original prompt version.
        new_prompt: Modified prompt version.
        embedder_mode: Embedding backend ('local' or 'openai').
        api_key: Optional OpenAI API key.

    Returns:
        Comprehensive drift report with per-section and overall metrics.

    Raises:
        ValueError: If embedder mode is invalid.
        ImportError: If required dependencies are missing.
    """
    embedder = EmbedderFactory.create(mode=embedder_mode, api_key=api_key)

    # Collect all section names from both prompts
    all_names = set(old_prompt.names()) | set(new_prompt.names())

    section_drifts: dict[str, SectionDrift] = {}
    drift_scores: list[float] = []

    for name in sorted(all_names):
        old_section = old_prompt.get(name)
        new_section = new_prompt.get(name)

        # Treat missing sections as empty
        if old_section is None:
            old_section = PromptSection(name=name, content="")
        if new_section is None:
            new_section = PromptSection(name=name, content="")

        section_drift = score_section(old_section, new_section, embedder)
        section_drifts[name] = section_drift
        drift_scores.append(section_drift.drift_score)

    # Calculate overall drift as mean
    overall_drift = float(np.mean(drift_scores)) if drift_scores else 0.0

    return DriftReport(
        sections=section_drifts,
        overall_drift=overall_drift,
    )
