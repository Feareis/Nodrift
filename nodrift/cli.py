"""Command-line interface for comparing prompt versions.

Provides a typer-based CLI with rich output for semantic drift analysis.
Supports both human-readable and JSON output formats.
"""

from __future__ import annotations

import json as json_lib
import sys
from pathlib import Path

import typer
from rich.console import Console

from nodrift.parser import PromptParseError, parse_file
from nodrift.reporter import format_report
from nodrift.scorer import diff as calculate_diff

# Global consoles for output
_console = Console()
_err_console = Console(stderr=True)

app = typer.Typer(
    name="nodrift",
    help="Semantic versioning for LLM prompts",
    no_args_is_help=True,
)


@app.command()
def diff(
    old: Path = typer.Argument(
        ...,
        help="Path to the original prompt file",
        metavar="OLD",
    ),
    new: Path = typer.Argument(
        ...,
        help="Path to the new prompt file",
        metavar="NEW",
    ),
    threshold: float = typer.Option(
        0.40,
        "--threshold",
        "-t",
        help="Drift threshold for 'breaking' classification (0.0-1.0)",
        min=0.0,
        max=1.0,
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as machine-readable JSON",
    ),
) -> None:
    """Compare two prompt versions and detect semantic drift.

    Reads two prompt files, compares their semantic content using embeddings,
    and produces a drift report. Exits with code 1 if drift exceeds threshold.

    Examples:
        nodrift diff v1.txt v2.txt
        nodrift diff v1.txt v2.txt --threshold 0.25
        nodrift diff v1.txt v2.txt --json
    """
    try:
        # Validate paths
        if not old.exists():
            raise FileNotFoundError(f"File not found: {old}")
        if not new.exists():
            raise FileNotFoundError(f"File not found: {new}")

        # Parse prompts
        old_prompt = parse_file(old)
        new_prompt = parse_file(new)

        # Calculate drift
        report = calculate_diff(old_prompt, new_prompt)

        # Output result
        if json_output:
            _output_json(report)
        else:
            _output_formatted(report, threshold)

        # Exit with appropriate code
        sys.exit(0 if report.overall_drift <= threshold else 1)

    except (FileNotFoundError, PromptParseError) as e:
        _err_console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        _err_console.print(f"[red]Unexpected error: {e}[/red]")
        sys.exit(2)


def _output_json(report) -> None:
    """Output drift report in JSON format."""
    output = {
        "overall_drift": round(report.overall_drift, 4),
        "overall_severity": report.overall_severity,
        "sections": {
            name: {
                "drift": round(section.drift_score, 4),
                "similarity": round(section.similarity, 4),
                "severity": section.severity,
            }
            for name, section in report.sections.items()
        },
    }
    _console.print(json_lib.dumps(output, indent=2))


def _output_formatted(report, threshold: float) -> None:
    """Output drift report in human-readable format."""
    format_report(_console, report, threshold)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Nodrift: Detect behavioral drift between prompt versions."""
    if ctx.invoked_subcommand is None:
        _console.print("[bold cyan]Nodrift v0.1.0[/bold cyan]")
        _console.print("Semantic versioning for LLM prompts\n")
        _console.print("[bold]Usage:[/bold]")
        _console.print("  [cyan]nodrift diff <old> <new>[/cyan]\n")
        _console.print("[bold]Options:[/bold]")
        _console.print("  -t, --threshold    Drift threshold (default: 0.40)")
        _console.print("  -j, --json         Output JSON format\n")
        _console.print("Run [bold]nodrift --help[/bold] for more information.")


if __name__ == "__main__":
    app()
