"""Typer-based CLI entry point.

Sub-commands:
* ``safediff compare A B``  — compare two checkpoints (the original diff)
* ``safediff track  <dir>`` — trace weight evolution across a checkpoint directory
* ``safediff audit  <file>``— static health check on a single checkpoint
* ``safediff demo``         — run a built-in demo
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from safediff import __version__
from safediff.analyzer import analyze
from safediff.audit import audit
from safediff.loader import load_tensors
from safediff.track import discover_checkpoints, track
from safediff.visualizer import render_audit, render_dead_neurons, render_json, render_report, render_track_summary

console = Console()
app = typer.Typer(
    name="safediff",
    help="Checkpoint static audit and learning dynamics tracking for PyTorch models.",
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
    """safediff — checkpoint audit and learning dynamics tracking."""


# ------------------------------------------------------------------------------------------
# compare (formerly diff)
# ------------------------------------------------------------------------------------------

@app.command(name="compare")
def compare_command(
    a: Path = typer.Argument(..., exists=True, readable=True, help="First checkpoint (A)."),
    b: Path = typer.Argument(..., exists=True, readable=True, help="Second checkpoint (B)."),
    eps: float = typer.Option(1e-6, "--eps", help="Dead-neuron threshold on |ΔW|."),
    top: int = typer.Option(20, "--top", min=1, help="Show only the top-N layers by L2 norm."),
    anomaly_threshold: float = typer.Option(
        10.0, "--anomaly-threshold", help="Flag a layer when L2 > median × threshold."
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
    """Compare two checkpoints and print a per-layer diff report.

    This is the original safediff command. Use ``track`` to trace weight evolution
    across multiple checkpoints over time.
    """
    try:
        with console.status("[cyan]Loading checkpoint A…[/cyan]"):
            tensors_a = load_tensors(a)
        with console.status("[cyan]Loading checkpoint B…[/cyan]"):
            tensors_b = load_tensors(b)
    except Exception as exc:
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


# Alias: "diff" still works for backwards compatibility
@app.command(name="diff", hidden=True)
def diff_command(
    a: Path = typer.Argument(..., exists=True, readable=True, help="First checkpoint."),
    b: Path = typer.Argument(..., exists=True, readable=True, help="Second checkpoint."),
    eps: float = typer.Option(1e-6, "--eps", help="Dead-neuron threshold."),
    top: int = typer.Option(20, "--top", min=1),
    anomaly_threshold: float = typer.Option(10.0, "--anomaly-threshold"),
    no_sparkline: bool = typer.Option(False, "--no-sparkline"),
    filter_pattern: Optional[str] = typer.Option(None, "--filter"),
    fmt: str = typer.Option("table", "--format"),
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
    sparkline_width: int = typer.Option(40, "--sparkline-width"),
    no_dead: bool = typer.Option(False, "--no-dead"),
) -> None:
    """Alias for ``compare``. Use ``compare`` for new scripts."""
    compare_command(
        a=a, b=b, eps=eps, top=top, anomaly_threshold=anomaly_threshold,
        no_sparkline=no_sparkline, filter_pattern=filter_pattern,
        fmt=fmt, output=output, sparkline_width=sparkline_width, no_dead=no_dead,
    )


# ------------------------------------------------------------------------------------------
# track
# ------------------------------------------------------------------------------------------

@app.command(name="track")
def track_command(
    directory: Path = typer.Argument(
        ..., exists=True, file_okay=False, dir_okay=True,
        help="Directory containing checkpoint files.",
    ),
    top: int = typer.Option(15, "--top", min=1, help="Show only the top-N layers by L2 drift."),
    metric: str = typer.Option(
        "cumulative_l2", "--metric",
        help="Drift metric: 'cumulative_l2' (from baseline) or 'incremental_l2' (per-step).",
    ),
    anomaly_threshold: float = typer.Option(
        3.5, "--anomaly-threshold",
        help="Modified z-score threshold for divergence detection (3.5 = NIST recommended).",
    ),
    dead_eps: float = typer.Option(1e-6, "--eps", help="Dead-neuron threshold on |ΔW|."),
    filter_pattern: Optional[str] = typer.Option(
        None, "--filter", help="Only track layers matching this glob pattern (e.g. '*.mlp.*')."
    ),
    fmt: str = typer.Option("table", "--format", help="Output format: table (default) or json."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write report to file."),
    output_dir: Optional[Path] = typer.Option(
        None, "--checkpoints-dir",
        help="Explicit checkpoint directory (overrides positional argument).",
    ),
) -> None:
    """Trace per-layer weight evolution across a sequence of checkpoints.

    Automatically detects which layer first started diverging, even before the
    overall loss spike became visible in training logs.

    Example:
        safediff track ./checkpoints --top 10
        safediff track ./checkpoints --filter "*.attn.*" --metric incremental_l2
    """
    target_dir = directory if directory.is_dir() else directory.parent

    try:
        checkpoints = discover_checkpoints(target_dir)
    except ValueError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    if len(checkpoints) < 2:
        console.print(
            f"[bold red]Error:[/bold red] Only {len(checkpoints)} checkpoint(s) found. "
            "Need at least 2 to track dynamics."
        )
        raise typer.Exit(code=1)

    with console.status(
        f"[cyan]Loading {len(checkpoints)} checkpoints…[/cyan]"
    ):
        layer_series, alerts = track(
            checkpoints,
            loader=load_tensors,
            dead_eps=dead_eps,
            anomaly_threshold=anomaly_threshold,
        )

    # Apply layer filter
    if filter_pattern:
        import fnmatch
        layer_series = {
            k: v for k, v in layer_series.items()
            if fnmatch.fnmatch(k, filter_pattern)
        }
        alerts = [a for a in alerts if fnmatch.fnmatch(a.layer_name, filter_pattern)]

    if not layer_series:
        console.print("[yellow]No layers match the filter.[/yellow]")
        raise typer.Exit()

    if fmt == "json":
        import json
        payload = {
            "checkpoints": [str(c.path) for c in checkpoints],
        }
        console.print(json.dumps(payload, indent=2))
        return

    capture: Console
    close_after = False
    if output:
        capture = Console(file=open(output, "w"), width=120, force_terminal=False)
        close_after = True
    else:
        capture = console

    try:
        render_track_summary(
            checkpoints,
            layer_series,
            alerts,
            console=capture,
            top=top,
            metric=metric,
        )
    finally:
        if close_after:
            capture.file.close()  # type: ignore[attr-defined]


# ------------------------------------------------------------------------------------------
# audit
# ------------------------------------------------------------------------------------------

@app.command(name="audit")
def audit_command(
    file: Path = typer.Argument(
        ..., exists=True, readable=True,
        help="Checkpoint file to audit (.safetensors / .pt / .pth / .bin).",
    ),
    outlier_sigma: float = typer.Option(
        5.0, "--outlier-sigma",
        help="Flag values beyond this many standard deviations as outliers.",
    ),
    near_zero_eps: float = typer.Option(
        1e-6, "--eps",
        help="Treat values with |x| < eps as near-zero.",
    ),
    top: int = typer.Option(10, "--top", min=1, help="Show only the top-N outlier layers."),
    fmt: str = typer.Option("table", "--format", help="Output format: table (default) or json."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write report to file."),
    no_frozen: bool = typer.Option(False, "--no-frozen", help="Skip frozen-layer detection."),
) -> None:
    """Run a static health check on a single checkpoint.

    Scans for:
    - NaN / Inf values (critical — will crash GPU kernels)
    - Extreme outliers (blocks INT8 / GPTQ / AWQ quantisation)
    - Near-zero layers (wasted compute / memory)
    - Frozen layer pairs (likely copy-paste bugs)

    Example:
        safediff audit model.safetensors
        safediff audit model.pt --outlier-sigma 4
    """
    try:
        with console.status("[cyan]Loading checkpoint…[/cyan]"):
            tensors = load_tensors(file)
    except Exception as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    from safediff.audit import audit as _audit_impl
    report = _audit_impl(
        tensors,
        outlier_sigma=outlier_sigma,
        near_zero_eps=near_zero_eps,
    )

    if fmt == "json":
        import json
        payload = {
            "path": str(file),
            "is_healthy": report.is_healthy,
            "critical_count": report.critical_count,
            "warning_count": report.warning_count,
            "nan_layers": [r.name for r in report.nan_layers],
            "inf_layers": [r.name for r in report.inf_layers],
            "outlier_layers": [
                {"name": r.name, "fraction": r.outlier_fraction}
                for r in report.outlier_layers
            ],
            "near_zero_layers": [
                {"name": r.name, "fraction": r.near_zero_fraction}
                for r in report.near_zero_layers
            ],
        }
        text = json.dumps(payload, indent=2)
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
        render_audit(report, console=capture, top_outliers=top)
    finally:
        if close_after:
            capture.file.close()  # type: ignore[attr-defined]


# ------------------------------------------------------------------------------------------
# demo
# ------------------------------------------------------------------------------------------

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
