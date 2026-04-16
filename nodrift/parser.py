"""Module for parsing system prompts into structured sections.

This module provides functionality to parse raw system prompts into
semantic sections. Sections can be explicitly delimited with [section_name]
headers, or the entire prompt can be treated as a single implicit section.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


DEFAULT_SECTION_NAME = "__default__"
SECTION_HEADER_PATTERN = re.compile(r"^\[([a-zA-Z0-9_-]+)\]\s*$", re.MULTILINE)


@dataclass(frozen=True)
class PromptSection:
    """An immutable section of a system prompt.

    Attributes:
        name: Unique identifier for this section (lowercase).
        content: Raw text content of the section.
    """

    name: str
    content: str

    @property
    def is_empty(self) -> bool:
        """Check if section has meaningful content."""
        return not bool(self.content.strip())

    def __bool__(self) -> bool:
        """Alias for is_empty check in boolean context."""
        return not self.is_empty


@dataclass(frozen=True)
class ParsedPrompt:
    """Immutable result of parsing a system prompt.

    Attributes:
        sections: Ordered list of parsed sections.
    """

    sections: tuple[PromptSection, ...] = ()

    def __iter__(self) -> Iterator[PromptSection]:
        """Iterate over sections in order."""
        return iter(self.sections)

    def __len__(self) -> int:
        """Return number of sections."""
        return len(self.sections)

    def __getitem__(self, index: int) -> PromptSection:
        """Access section by index."""
        return self.sections[index]

    def get(self, name: str) -> PromptSection | None:
        """Retrieve section by name (case-insensitive lookup).

        Args:
            name: Section name to look up.

        Returns:
            Section if found, None otherwise.
        """
        for section in self.sections:
            if section.name == name.lower():
                return section
        return None

    def names(self) -> list[str]:
        """Get ordered list of section names."""
        return [s.name for s in self.sections]

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        section_list = ", ".join(self.names())
        return f"ParsedPrompt(sections=[{section_list}])"


class PromptParseError(ValueError):
    """Raised when prompt parsing fails."""

    pass


def parse(text: str) -> ParsedPrompt:
    """Parse raw prompt text into structured sections.

    Sections are delimited by [section_name] headers. If no headers are
    found, the entire text is treated as a single "__default__" section.
    Text before the first header is also treated as "__default__".

    Args:
        text: Raw prompt text to parse.

    Returns:
        ParsedPrompt containing parsed sections.

    Example:
        >>> p = parse("[tone]\\nBe friendly.\\n[rules]\\nNo profanity.")
        >>> p.names()
        ['tone', 'rules']
    """
    text = text.strip()
    if not text:
        return ParsedPrompt()

    matches = list(SECTION_HEADER_PATTERN.finditer(text))

    if not matches:
        return ParsedPrompt(
            sections=(PromptSection(name=DEFAULT_SECTION_NAME, content=text),)
        )

    sections: list[PromptSection] = []

    # Handle preamble before first section header
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(PromptSection(name=DEFAULT_SECTION_NAME, content=preamble))

    # Parse explicit sections
    for i, match in enumerate(matches):
        section_name = match.group(1).lower()
        section_start = match.end()
        section_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_content = text[section_start:section_end].strip()

        sections.append(PromptSection(name=section_name, content=section_content))

    return ParsedPrompt(sections=tuple(sections))


def parse_file(path: str | Path) -> ParsedPrompt:
    """Load and parse a prompt from a file.

    Args:
        path: Path to the prompt file.

    Returns:
        ParsedPrompt from file contents.

    Raises:
        FileNotFoundError: If file does not exist.
        PromptParseError: If file encoding is invalid.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        raise PromptParseError(f"File {path} is not valid UTF-8: {e}") from e

    return parse(content)
