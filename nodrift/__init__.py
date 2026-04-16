"""
Nodrift — semantic versioning for LLM prompts.
Detect behavioral drift between two versions of a system prompt.
"""

from nodrift.parser import parse, parse_file
from nodrift.scorer import diff

__all__ = ["parse", "parse_file", "diff"]
__version__ = "0.1.0"
