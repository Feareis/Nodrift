"""Formatted output generation for drift reports.

Provides rich, colored terminal output and potential future support for
HTML/PDF report generation.
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from nodrift.scorer import DriftReport


def format_report(
    console: Console,
    report: DriftReport,
    threshold: float = 0.40,
) -> None:
    """Display drift report with color-coded severity levels.

    Produces a formatted table showing per-section drift metrics and overall
    summary with actionable recommendations based on drift threshold.

    Args:
        console: Rich Console instance for output.
        report: DriftReport to display.
        threshold: Drift value that triggers 'breaking' classification.
    """
    _print_header(console)
    _print_section_table(console, report)
    _print_summary(console, report, threshold)


def _print_header(console: Console) -> None:
    """Print report header."""
    console.print("\n[bold cyan]Semantic Drift Report[/bold cyan]")
    console.print("─" * 60)


def _print_section_table(console: Console, report: DriftReport) -> None:
    """Print per-section drift metrics in table format."""
    table = Table(
        title="Section Analysis",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Section", style="cyan", width=20)
    table.add_column("Drift", justify="right", width=12)
    table.add_column("Status", justify="center", width=15)

    for name, section in sorted(report.sections.items()):
        # Color and symbol based on severity
        severity_display = {
            "ok": ("[green]✓ OK[/green]", "green"),
            "warning": ("[yellow]⚠ Warning[/yellow]", "yellow"),
            "breaking": ("[red]✗ Breaking[/red]", "red"),
        }

        status_str, _color = severity_display[section.severity]

        table.add_row(
            name,
            f"{section.drift_score:.1%}",
            status_str,
        )

    console.print(table)


def _print_summary(console: Console, report: DriftReport, threshold: float) -> None:
    """Print overall drift summary and recommendations."""
    # Determine summary color
    summary_color = {
        "ok": "green",
        "warning": "yellow",
        "breaking": "red",
    }[report.overall_severity]

    # Print metrics
    console.print(f"\n[bold {summary_color}]Overall Drift: {report.overall_drift:.1%}[/]")
    console.print(f"[bold]Severity: {report.overall_severity.upper()}[/bold]")

    # Print recommendation
    if report.overall_drift > threshold:
        console.print(
            f"\n[bold red]Exceeds threshold ({threshold:.0%})[/bold red]\n"
            "[yellow]→ Review changes before deploying[/yellow]\n"
        )
    else:
        console.print(
            f"\n[bold green]Within acceptable threshold ({threshold:.0%})[/bold green]\n"
        )
