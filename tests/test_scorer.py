"""Tests pour nodrift.scorer."""

import numpy as np
import pytest

from nodrift.parser import ParsedPrompt, PromptSection, parse
from nodrift.scorer import (
    DriftReport,
    SectionDrift,
    cosine_similarity,
    diff,
    score_section,
    similarity_to_drift,
)


# ------------------------------------------------------------------ #
# Similarité cosinus                                                  #
# ------------------------------------------------------------------ #


def test_cosine_similarity_identical_vectors():
    """Vecteurs identiques → similarité 1.0"""
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    sim = cosine_similarity(a, b)
    assert np.isclose(sim, 1.0)


def test_cosine_similarity_orthogonal_vectors():
    """Vecteurs orthogonaux → similarité 0.0"""
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    sim = cosine_similarity(a, b)
    assert np.isclose(sim, 0.0)


def test_cosine_similarity_opposite_vectors():
    """Vecteurs opposés → similarité -1.0"""
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([-1.0, 0.0, 0.0])
    sim = cosine_similarity(a, b)
    assert np.isclose(sim, -1.0)


def test_cosine_similarity_zero_vector_returns_zero():
    """Vecteur zéro → similarité 0.0"""
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 1.0, 1.0])
    sim = cosine_similarity(a, b)
    assert sim == 0.0


# ------------------------------------------------------------------ #
# Conversion similarité → drift                                       #
# ------------------------------------------------------------------ #


def test_similarity_to_drift_identical():
    """Similarité 1.0 → drift 0.0"""
    drift = similarity_to_drift(1.0)
    assert drift == 0.0


def test_similarity_to_drift_orthogonal():
    """Similarité 0.0 → drift 0.5"""
    drift = similarity_to_drift(0.0)
    assert drift == 0.5


def test_similarity_to_drift_opposite():
    """Similarité -1.0 → drift 1.0"""
    drift = similarity_to_drift(-1.0)
    assert drift == 1.0


# ------------------------------------------------------------------ #
# SectionDrift severity                                               #
# ------------------------------------------------------------------ #


def test_section_drift_ok_severity():
    """drift 0.05 → OK"""
    section = SectionDrift(
        name="test",
        drift_score=0.05,
        similarity=0.9,
    )
    assert section.severity == "ok"


def test_section_drift_warning_severity():
    """drift 0.20 → Warning"""
    section = SectionDrift(
        name="test",
        drift_score=0.20,
        similarity=0.6,
    )
    assert section.severity == "warning"


def test_section_drift_breaking_severity():
    """drift 0.50 → Breaking"""
    section = SectionDrift(
        name="test",
        drift_score=0.50,
        similarity=0.0,
    )
    assert section.severity == "breaking"


# ------------------------------------------------------------------ #
# score_section()                                                     #
# ------------------------------------------------------------------ #


def test_score_section_both_empty():
    """Deux sections vides → drift 0.0 (OK)"""
    from nodrift.embedder import LocalEmbedder

    embedder = LocalEmbedder()
    old = PromptSection(name="test", content="")
    new = PromptSection(name="test", content="")

    result = score_section(old, new, embedder)
    assert result.drift_score == 0.0
    assert result.similarity == 1.0
    assert result.severity == "ok"


def test_score_section_one_empty():
    """Une section vide, l'autre pas → drift 1.0 (Breaking)"""
    from nodrift.embedder import LocalEmbedder

    embedder = LocalEmbedder()
    old = PromptSection(name="test", content="Be friendly")
    new = PromptSection(name="test", content="")

    result = score_section(old, new, embedder)
    assert result.drift_score == 1.0
    assert result.severity == "breaking"


def test_score_section_identical_text():
    """Textes identiques → drift bas (OK)"""
    from nodrift.embedder import LocalEmbedder

    embedder = LocalEmbedder()
    text = "Be friendly and helpful"
    old = PromptSection(name="tone", content=text)
    new = PromptSection(name="tone", content=text)

    result = score_section(old, new, embedder)
    assert result.drift_score < 0.15
    assert result.severity == "ok"


def test_score_section_very_different_text():
    """Textes très différents → drift élevé (Breaking)"""
    from nodrift.embedder import LocalEmbedder

    embedder = LocalEmbedder()
    old = PromptSection(name="tone", content="Be extremely polite and formal")
    new = PromptSection(name="tone", content="Machine learning algorithms quantum computing")

    result = score_section(old, new, embedder)
    assert result.drift_score > 0.40
    assert result.severity == "breaking"


# ------------------------------------------------------------------ #
# diff()                                                              #
# ------------------------------------------------------------------ #


def test_diff_identical_prompts():
    """Prompts identiques → drift global bas"""
    p1 = parse("[tone]\nBe friendly.\n[rules]\nNo refunds.")
    p2 = parse("[tone]\nBe friendly.\n[rules]\nNo refunds.")

    report = diff(p1, p2)
    assert report.overall_drift < 0.15
    assert report.overall_severity == "ok"


def test_diff_returns_drift_report():
    """diff() retourne un DriftReport valide."""
    p1 = parse("[tone]\nBe friendly.")
    p2 = parse("[tone]\nBe friendly.")

    result = diff(p1, p2)
    assert isinstance(result, DriftReport)
    assert hasattr(result, "sections")
    assert hasattr(result, "overall_drift")
    # Note: floating point precision may produce slightly negative values
    assert -1e-6 <= result.overall_drift <= 1


def test_diff_new_section_added():
    """Section ajoutée dans p2 → drift détecté"""
    p1 = parse("[tone]\nBe friendly.")
    p2 = parse("[tone]\nBe friendly.\n[rules]\nNo refunds.")

    report = diff(p1, p2)
    assert "rules" in report.sections
    assert report.sections["rules"].drift_score == 1.0  # Section créée
    assert report.sections["rules"].severity == "breaking"


def test_diff_section_removed():
    """Section supprimée dans p2 → drift détecté"""
    p1 = parse("[tone]\nBe friendly.\n[rules]\nNo refunds.")
    p2 = parse("[tone]\nBe friendly.")

    report = diff(p1, p2)
    assert "rules" in report.sections
    assert report.sections["rules"].drift_score == 1.0  # Section supprimée
    assert report.sections["rules"].severity == "breaking"


def test_diff_multiple_sections():
    """Différentes sévérités par section."""
    p1 = parse(
        "[tone]\nBe friendly.\n"
        "[rules]\nNo refunds without approval.\n"
        "[escalation]\nTransfer to manager."
    )
    p2 = parse(
        "[tone]\nBe friendly.\n"  # Inchangé
        "[rules]\nCryptographic algorithms and blockchain technology.\n"  # Très différent
        "[escalation]\n"  # Supprimé
    )

    report = diff(p1, p2)
    assert "tone" in report.sections
    assert "rules" in report.sections
    assert "escalation" in report.sections

    # tone: peu de changement
    assert report.sections["tone"].drift_score < 0.15

    # rules: très différent
    assert report.sections["rules"].drift_score > 0.40

    # escalation: supprimé
    assert report.sections["escalation"].drift_score == 1.0


def test_diff_overall_drift_is_mean():
    """Le drift global est la moyenne des drifts par section."""
    p1 = parse("[a]\nContent A\n[b]\nContent B")
    p2 = parse("[a]\nContent A\n[b]\nContent B")

    report = diff(p1, p2)
    # Tous les sections identiques → drift ~0.0 (with floating point tolerance)
    assert abs(report.overall_drift) < 1e-6


def test_diff_severity_classification():
    """DriftReport calcule correctement overall_severity."""
    p1 = parse("[tone]\nBe friendly")
    p2 = parse("[tone]\nBe friendly")

    report = diff(p1, p2)
    assert report.overall_severity == "ok"
