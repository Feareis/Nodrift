# NoDrift

[![Tests](https://img.shields.io/badge/tests-53%2F53-green)](./tests)
[![Coverage](https://img.shields.io/badge/coverage-67%25-green)](./tests)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-black)](./LICENSE)

Semantic versioning for LLM prompts.

NoDrift helps teams detect **behavioral drift** between two versions of a prompt before shipping changes to production. Instead of comparing only text differences, it compares semantic meaning and estimates how much behavior may have changed.

---

## Why this project exists

Prompt edits are easy to make and hard to validate:

- A tiny rewrite can alter model behavior.
- Traditional text diff shows what changed, not what those changes mean.
- Manual review is slow and inconsistent.

NoDrift addresses this by computing semantic similarity section by section and producing a drift report with actionable severity levels (`ok`, `warning`, `breaking`).

---

## What NoDrift does

Given two prompt files, NoDrift:

1. Parses them into sections (`[tone]`, `[rules]`, etc.).
2. Generates embeddings for each section (local or OpenAI backend).
3. Computes cosine similarity between old/new section vectors.
4. Converts similarity into a normalized drift score.
5. Produces a terminal or JSON report with per-section and overall severity.

This gives a practical signal for deciding whether a prompt change is safe to deploy.

---

## How prompt files are structured

NoDrift supports two formats:

### 1) Single-block prompt (implicit default section)

```txt
You are a customer support assistant.
Always be concise.
```

### 2) Sectioned prompt (recommended)

```txt
[tone]
Be calm and professional.

[rules]
Never invent refunds.

[escalation]
Escalate billing disputes to a manager.
```

Rules:

- Section headers use `[section-name]` syntax.
- Header names are normalized to lowercase.
- Text before the first header is stored as a special `__default__` section.

---

## Installation

### From source

```bash
git clone https://github.com/Feareis/nodrift.git
cd nodrift
pip install -e .
```

### Optional dependencies

- OpenAI backend support:

```bash
pip install -e .[openai]
```

- Development dependencies:

```bash
pip install -e .[dev]
```

---

## Quick start (CLI)

Compare two prompt versions:

```bash
nodrift diff tests/sample/v1.txt tests/sample/v2.txt
```

Use a custom failure threshold:

```bash
nodrift diff old.txt new.txt --threshold 0.30
```

Output machine-readable JSON:

```bash
nodrift diff old.txt new.txt --json
```

### Exit codes

- `0`: overall drift is within threshold.
- `1`: overall drift exceeds threshold, or input/parse error.
- `2`: unexpected runtime error.

This makes the CLI suitable for CI checks and pre-merge validation.

---

## Severity model

NoDrift computes drift scores from semantic similarity and maps them to severity bands:

- `ok`: drift `< 0.15`
- `warning`: drift `>= 0.15` and `< 0.40`
- `breaking`: drift `>= 0.40`

Overall drift is the mean of all section drift scores.

---

## Python API

You can integrate NoDrift in scripts and test pipelines:

```python
from nodrift import parse_file, diff

old_prompt = parse_file("v1.txt")
new_prompt = parse_file("v2.txt")

report = diff(old_prompt, new_prompt)

print(report.overall_drift)
print(report.overall_severity)
for name, section in report.sections.items():
    print(name, section.drift_score, section.severity)
```

---

## Embedding backends

NoDrift currently supports:

- `local` (default): `sentence-transformers/all-MiniLM-L6-v2`
- `openai`: `text-embedding-3-small`

By default, CLI uses the local backend (no external API required).

If using OpenAI in Python code, provide `OPENAI_API_KEY` or pass `api_key` when creating the embedder backend.

---

## Project architecture

Core modules:

- `nodrift/parser.py`: parses raw prompt files into structured sections.
- `nodrift/embedder.py`: embedding interfaces and backend factory.
- `nodrift/scorer.py`: similarity, drift scoring, and final report model.
- `nodrift/reporter.py`: rich terminal output formatting.
- `nodrift/cli.py`: `nodrift` Typer CLI command entry point.

This separation keeps parsing, semantic scoring, and presentation independent.

---

## Intended use cases

NoDrift is designed for teams that:

- ship prompts in production workflows,
- need review gates before deployment,
- want CI checks on prompt changes,
- maintain multiple versions of critical system prompts.

Typical scenarios include customer support bots, workflow assistants, compliance-sensitive agents, and prompt-heavy product teams.

---

## Current status

This project is currently in **alpha** (`0.1.0`).

Stable and available now:

- section parsing,
- semantic section scoring,
- overall drift classification,
- CLI text and JSON outputs,
- unit tests for parser/embedder/scorer behavior.

Planned next steps (roadmap direction):

- configurable per-section thresholds,
- golden drift tests,
- CI/GitHub Action integration,
- richer export formats.

---

## Running tests

```bash
pytest
```

Or run a subset:

```bash
pytest tests/test_parser.py
```

---

## License

MIT License. See [`LICENSE`](LICENSE).
