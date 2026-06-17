"""Terminal visualization: rich tables + Unicode sparklines."""

from __future__ import annotations

import json
import shutil

import numpy as np
from rich.console import Console
from rich.table import Table

from safediff.analyzer import DiffReport, LayerStat

# Eight block characters from low to high, used for sparklines and bars.
_BLOCKS = " ▂▃▄▅▆▇█"
_BAR_FILLED = "█"
_BAR_EMPTY = "░"


def _term_width(default: int = 80) -> int:
    try:
        return max(40, shutil.get_terminal_size().columns)
    except (OSError, ValueError):
        return default


def sparkline(values: np.ndarray, width: int = 40) -> str:
    """Return a sparkline of ``values`` drawn with Unicode block characters.

    The function bins ``values`` into ``width`` bins and maps each bin height
    onto one of 8 block glyphs. Empty input yields an empty string.
    """
    arr = np.asarray(values).ravel()
    if arr.size == 0 or width <= 0:
        return ""
    hist, _ = np.histogram(arr, bins=width)
    peak = hist.max()
    if peak == 0:
        return " " * width
    norm = hist / peak
    return "".join(_BLOCKS[min(int(v * 8), 7)] for v in norm)


def _bar(value: float, peak: float, width: int = 20) -> str:
    if peak <= 0 or value <= 0:
        return _BAR_EMPTY * width
    filled = int(round(width * value / peak))
    filled = max(0, min(width, filled))
    return _BAR_FILLED * filled + _BAR_EMPTY * (width - filled)


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: max(0, n - 1)] + "…"


def _stat_cell(stat: LayerStat) -> str:
    if stat.l2_norm == float("inf"):
        return "[bold red]shape mismatch[/bold red]"
    if not np.isfinite(stat.l2_norm):
        return "n/a"
    return f"{stat.max_abs:.3e}"


def render_report(
    report: DiffReport,
    *,
    console: Console | None = None,
    sparkline_width: int = 40,
    show_sparkline: bool = True,
    deltas: dict[str, np.ndarray] | None = None,
) -> None:
    """Render ``report`` to ``console`` (or stdout)."""
    console = console or Console()

    header = (
        f"[bold]safediff[/bold] "
        f"A={report.total_params_a:,} params, "
        f"B={report.total_params_b:,} params, "
        f"common={len(report.common)} layers"
    )
    console.print(header)

    table = Table(
        show_header=True,
        header_style="bold",
        title="Per-layer diff (sorted by L2 norm, descending)",
        title_style="bold cyan",
    )
    table.add_column("Layer", style="cyan", no_wrap=True)
    table.add_column("Shape", justify="right")
    table.add_column("max|ΔW|", justify="right")
    table.add_column("L2 norm", justify="right")
    table.add_column("mean", justify="right")
    table.add_column("std", justify="right")
    table.add_column("dead %", justify="right")
    table.add_column("ΔW distribution", ratio=1)

    if not report.common:
        console.print("[yellow]No common layers between A and B.[/yellow]")
        return

    # L2 peak for the bar column.
    finite_l2s = [s.l2_norm for s in report.common if np.isfinite(s.l2_norm)]
    l2_peak = max(finite_l2s) if finite_l2s else 0.0

    width = console.width or _term_width()
    name_w = max(20, min(width - 60, 60))
    sw_w = sparkline_width if show_sparkline else 0

    for stat in report.common:
        row_style = "bold red" if stat.is_anomaly else None
        name = _truncate(stat.name, name_w)
        if stat.l2_norm == float("inf"):
            cells = [name, str(stat.shape), "∞", "∞", "-", "-", "-", ""]
        else:
            bar = _bar(stat.l2_norm, l2_peak, width=20)
            dead = f"{stat.is_dead_fraction * 100:5.2f}%"
            spark = (
                sparkline(
                    deltas.get(stat.name, np.array([])),  # type: ignore[arg-type]
                    width=sw_w,
                )
                if show_sparkline and deltas is not None
                else ""
            )
            cells = [
                name,
                "x".join(str(s) for s in stat.shape),
                f"{stat.max_abs:.3e}",
                f"{stat.l2_norm:.3e} {bar}",
                f"{stat.mean:+.3e}",
                f"{stat.std:.3e}",
                dead,
                spark,
            ]
        table.add_row(*cells, style=row_style)

    console.print(table)

    if report.anomalies:
        console.print(
            f"\n[bold red]⚠  {len(report.anomalies)} anomalous layer(s) detected[/bold red]"
        )

    if report.only_in_a:
        console.print(
            f"\n[dim]{len(report.only_in_a)} key(s) only in A:[/dim] "
            + ", ".join(_truncate(k, 40) for k in report.only_in_a[:5])
            + (" ..." if len(report.only_in_a) > 5 else "")
        )
    if report.only_in_b:
        console.print(
            f"[dim]{len(report.only_in_b)} key(s) only in B:[/dim] "
            + ", ".join(_truncate(k, 40) for k in report.only_in_b[:5])
            + (" ..." if len(report.only_in_b) > 5 else "")
        )


def render_dead_neurons(
    report: DiffReport,
    *,
    console: Console | None = None,
    top: int = 20,
) -> None:
    """Render a separate table ranking layers by dead-ration."""
    console = console or Console()
    table = Table(
        show_header=True,
        header_style="bold",
        title=f"Top {top} layers by dead-parameter ratio",
        title_style="bold yellow",
    )
    table.add_column("Layer", style="yellow", no_wrap=True)
    table.add_column("dead", justify="right")
    table.add_column("total", justify="right")
    table.add_column("ratio", justify="right")
    table.add_column("distribution", ratio=1)

    candidates = [s for s in report.common if s.is_dead_fraction > 0.0][:top]
    if not candidates:
        console.print("[green]No dead parameters detected at the current threshold.[/green]")
        return

    peak = max(c.is_dead_fraction for c in candidates) or 1.0
    for s in candidates:
        bar = _bar(s.is_dead_fraction, peak, width=20)
        table.add_row(
            _truncate(s.name, 60),
            f"{int(s.is_dead_fraction * s.numel):,}",
            f"{s.numel:,}",
            f"{s.is_dead_fraction * 100:6.2f}%",
            bar,
        )
    console.print(table)


def render_json(
    report: DiffReport, *, deltas: dict[str, np.ndarray] | None = None
) -> str:
    """Serialize a report to JSON. Deltas are NOT embedded to keep the output small."""
    payload = {
        "totals": {
            "params_a": report.total_params_a,
            "params_b": report.total_params_b,
            "common_layers": len(report.common),
        },
        "anomalies": [s.name for s in report.anomalies],
        "only_in_a": report.only_in_a,
        "only_in_b": report.only_in_b,
        "layers": [s.to_dict() for s in report.common],
    }
    return json.dumps(payload, indent=2)
