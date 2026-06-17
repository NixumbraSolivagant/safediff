"""Typer-based CLI entry point.

Sub-commands:
* ``safediff diff A B [OPTIONS]`` — compare two checkpoints
* ``safediff demo [OPTIONS]`` — run a built-in demo
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from safediff import __version__
from safediff.analyzer import analyze
from safediff.loader import load_tensors
from safediff.visualizer import render_dead_neurons, render_json, render_report

console = Console()
app = typer.Typer(
    name="safediff",
    help="Diff PyTorch model weights like you diff code.",
    add_completion=False,
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"safediff {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show safediff version and exit.",
    ),
) -> None:
    """safediff — analyze the difference between two model checkpoints."""


def _run_diff(
    a: Path,
    b: Path,
    eps: float,
    top: int,
    anomaly_threshold: float,
    no_sparkline: bool,
    filter_pattern: Optional[str],
    fmt: str,
    output: Optional[Path],
    sparkline_width: int,
    no_dead: bool,
) -> None:
    try:
        with console.status("[cyan]Loading checkpoint A…[/cyan]"):
            tensors_a = load_tensors(a)
        with console.status("[cyan]Loading checkpoint B…[/cyan]"):
            tensors_b = load_tensors(b)
    except Exception as exc:  # surface loader errors with a clear message
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    report = analyze(
        tensors_a,
        tensors_b,
        dead_eps=eps,
        anomaly_threshold=anomaly_threshold,
    )
    if filter_pattern:
        report.filter(filter_pattern)
    report.head(top)

    if fmt == "json":
        text = render_json(report)
        if output:
            output.write_text(text)
        else:
            console.print(text)
        return

    capture: Console
    close_after = False
    if output:
        capture = Console(file=open(output, "w"), width=120, force_terminal=False)
        close_after = True
    else:
        capture = console

    try:
        deltas = {
            name: tensors_b[name] - tensors_a[name]
            for name in {s.name for s in report.common}
            if name in tensors_a and name in tensors_b
        }
        render_report(
            report,
            console=capture,
            sparkline_width=sparkline_width,
            show_sparkline=not no_sparkline,
            deltas=deltas,
        )
        if not no_dead:
            render_dead_neurons(report, console=capture, top=min(top, 20))
    finally:
        if close_after:
            capture.file.close()  # type: ignore[attr-defined]


@app.command(name="diff")
def diff_command(
    a: Path = typer.Argument(..., exists=True, readable=True, help="First checkpoint (A)."),
    b: Path = typer.Argument(..., exists=True, readable=True, help="Second checkpoint (B)."),
    eps: float = typer.Option(1e-6, "--eps", help="Dead-neuron threshold on |ΔW|."),
    top: int = typer.Option(20, "--top", min=1, help="Show only the top-N layers by L2 norm."),
    anomaly_threshold: float = typer.Option(
        10.0, "--anomaly-threshold", help="Flag a layer when L2 > median * threshold."
    ),
    no_sparkline: bool = typer.Option(False, "--no-sparkline", help="Disable sparkline column."),
    filter_pattern: Optional[str] = typer.Option(
        None, "--filter", help="Glob to keep only matching layer names (e.g. '*.attn.*')."
    ),
    fmt: str = typer.Option("table", "--format", help="Output format: table (default) or json."),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Write the report to this file instead of stdout."
    ),
    sparkline_width: int = typer.Option(
        40, "--sparkline-width", min=8, max=120, help="Sparkline column width."
    ),
    no_dead: bool = typer.Option(False, "--no-dead", help="Skip the dead-neuron table."),
) -> None:
    """Compare two checkpoints and print a per-layer diff report."""
    _run_diff(
        a,
        b,
        eps,
        top,
        anomaly_threshold,
        no_sparkline,
        filter_pattern,
        fmt,
        output,
        sparkline_width,
        no_dead,
    )


@app.command(name="demo")
def demo_command(
    anomaly_threshold: float = typer.Option(8.0, "--anomaly-threshold"),
    top: int = typer.Option(15, "--top"),
    fmt: str = typer.Option("table", "--format"),
) -> None:
    """Run a built-in demo (downloads a tiny GPT-2 on first use)."""
    from safediff.demo import run_demo

    try:
        report, deltas = run_demo()
    except ImportError as exc:
        console.print(
            f"[bold red]Error:[/bold red] {exc}\n"
            f"Install the demo extra with: pip install 'safediff[demo]'"
        )
        raise typer.Exit(code=1) from exc
    report.head(top)
    if fmt == "json":
        console.print(render_json(report))
        return
    render_report(report, sparkline_width=40, deltas=deltas)
    render_dead_neurons(report)


if __name__ == "__main__":
    app()  # pragma: no cover
