"""Tests unitaires pour nodrift.parser."""

from pathlib import Path

import pytest

from nodrift.parser import (
    DEFAULT_SECTION_NAME,
    ParsedPrompt,
    PromptSection,
    parse,
    parse_file,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ------------------------------------------------------------------ #
# parse() — cas de base                                               #
# ------------------------------------------------------------------ #


def test_parse_empty_string_returns_empty():
    result = parse("")
    assert len(result) == 0
    assert result.sections == ()


def test_parse_whitespace_only_returns_empty():
    result = parse("   \n\n  ")
    assert len(result) == 0


def test_parse_no_sections_returns_default():
    text = "You are a helpful assistant."
    result = parse(text)
    assert result.names() == [DEFAULT_SECTION_NAME]
    assert result.get(DEFAULT_SECTION_NAME).content == text


def test_parse_multiline_no_sections():
    text = "Line one.\nLine two.\nLine three."
    result = parse(text)
    assert len(result) == 1
    assert result.get(DEFAULT_SECTION_NAME).content == text


# ------------------------------------------------------------------ #
# parse() — sections explicites                                       #
# ------------------------------------------------------------------ #


def test_parse_single_section():
    text = "[tone]\nBe friendly."
    result = parse(text)
    assert result.names() == ["tone"]
    assert result.get("tone").content == "Be friendly."


def test_parse_multiple_sections():
    text = "[tone]\nBe friendly.\n\n[rules]\nNo refunds."
    result = parse(text)
    assert result.names() == ["tone", "rules"]
    assert result.get("tone").content == "Be friendly."
    assert result.get("rules").content == "No refunds."


def test_parse_section_names_lowercased():
    text = "[TONE]\nBe friendly.\n[Rules]\nNo refunds."
    result = parse(text)
    assert result.names() == ["tone", "rules"]


def test_parse_section_with_hyphens_and_underscores():
    text = "[output-format]\nJSON only.\n[my_rules]\nBe safe."
    result = parse(text)
    assert "output-format" in result.names()
    assert "my_rules" in result.names()


def test_parse_with_preamble_creates_DEFAULT_SECTION_NAME():
    text = "Preamble content.\n\n[section]\nSection content."
    result = parse(text)
    assert result.names() == [DEFAULT_SECTION_NAME, "section"]
    assert result.get(DEFAULT_SECTION_NAME).content == "Preamble content."
    assert result.get("section").content == "Section content."


def test_parse_section_content_is_stripped():
    text = "[tone]\n\n   Be friendly.   \n\n"
    result = parse(text)
    assert result.get("tone").content == "Be friendly."


# ------------------------------------------------------------------ #
# ParsedPrompt helpers                                                 #
# ------------------------------------------------------------------ #


def test_get_missing_section_returns_none():
    result = parse("[tone]\nBe friendly.")
    assert result.get("missing") is None


def test_iteration_yields_sections():
    text = "[a]\nContent A.\n[b]\nContent B."
    result = parse(text)
    names = [s.name for s in result]
    assert names == ["a", "b"]


def test_len_counts_sections():
    text = "[a]\nA.\n[b]\nB.\n[c]\nC."
    result = parse(text)
    assert len(result) == 3


def test_repr_contains_section_names():
    result = parse("[tone]\nA.\n[rules]\nB.")
    r = repr(result)
    assert "tone" in r
    assert "rules" in r


# ------------------------------------------------------------------ #
# PromptSection bool                                                   #
# ------------------------------------------------------------------ #


def test_section_bool_false_when_empty_content():
    s = PromptSection(name="tone", content="   ")
    assert not bool(s)


def test_section_bool_true_when_has_content():
    s = PromptSection(name="tone", content="Be friendly.")
    assert bool(s)


# ------------------------------------------------------------------ #
# parse_file()                                                         #
# ------------------------------------------------------------------ #


def test_parse_file_simple(tmp_path):
    f = tmp_path / "prompt.txt"
    f.write_text("You are helpful.", encoding="utf-8")
    result = parse_file(f)
    assert result.get(DEFAULT_SECTION_NAME).content == "You are helpful."


def test_parse_file_sectioned():
    result = parse_file(FIXTURES / "sectioned.txt")
    assert "tone" in result.names()
    assert "rules" in result.names()
    assert "escalation" in result.names()


def test_parse_file_simple_fixture():
    result = parse_file(FIXTURES / "simple.txt")
    assert len(result) == 1
    assert result.get(DEFAULT_SECTION_NAME) is not None


def test_parse_file_with_preamble_fixture():
    result = parse_file(FIXTURES / "with_preamble.txt")
    assert DEFAULT_SECTION_NAME in result.names()
    assert "output-format" in result.names()
    assert "constraints" in result.names()


def test_parse_file_not_found_raises():
    with pytest.raises(FileNotFoundError, match="not found"):
        parse_file("/nonexistent/path/prompt.txt")


def test_parse_file_invalid_utf8_raises(tmp_path):
    f = tmp_path / "bad.txt"
    f.write_bytes(b"\xff\xfe invalid bytes")
    with pytest.raises(ValueError, match="UTF-8"):
        parse_file(f)
