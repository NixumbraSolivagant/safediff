"""Terminal visualization: rich tables + Unicode sparklines."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

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


# ------------------------------------------------------------------------------------------
# Learning Dynamics Tracker rendering
# ------------------------------------------------------------------------------------------

from safediff.track import (
    CheckpointInfo,
    DivergenceAlert,
    LayerSeries,
    top_layers_by_drift,
    normalize,
)


def _draw_trend_line(values: list[float], width: int = 30) -> str:
    """Draw a Unicode bar-chart column from a list of values (first → last)."""
    if len(values) < 2:
        return " " * width
    normed = normalize(values)
    col_h = 6  # characters of vertical resolution
    if width <= 0:
        return ""
    # Build column-major output: for each row (top to bottom), output one
    # block char per column (left to right) if the column's "height" is at or
    # above that row.
    lines = [""] * col_h
    for x_idx, norm_val in enumerate(normed):
        if norm_val <= 0:
            continue
        # Invert so first value is left, last is right; taller = lower row
        row_idx = int((1.0 - norm_val) * (col_h - 1))
        row_idx = max(0, min(col_h - 1, row_idx))
        # Mark this column up to its row_idx
        # We'll iterate over columns and pad; here we just record (x_idx, row_idx)
        # Build output as: each row string padded to x_idx+1 width, with a block
        for r in range(col_h):
            if r >= row_idx:
                lines[r] = lines[r].ljust(x_idx) + "▄"
            else:
                lines[r] = lines[r].ljust(x_idx + 1)
    return "\n".join(lines)


def render_track_summary(
    checkpoints: list[CheckpointInfo],
    layer_series: dict[str, LayerSeries],
    alerts: list[DivergenceAlert],
    *,
    console: Console | None = None,
    top: int = 15,
    metric: str = "cumulative_l2",
) -> None:
    """Render the Learning Dynamics summary table to the terminal."""
    console = console or Console()

    n_ckpts = len(checkpoints)
    labels = [c.label for c in checkpoints]

    console.print(
        f"[bold]safediff track[/bold]  {len(layer_series)} layers × {n_ckpts} checkpoints  "
        f"({checkpoints[0].label} → {checkpoints[-1].label})"
    )

    # --- Divergence alerts ---
    if alerts:
        console.print(f"\n[bold red]⚠  {len(alerts)} layer(s) diverged during training[/bold red]")
        for alert in alerts[:5]:
            step_label = (
                checkpoints[alert.first_drift_step].label
                if alert.first_drift_step < len(checkpoints)
                else str(alert.first_drift_step)
            )
            console.print(
                f"  [red]▸[/red]  [cyan]{alert.layer_name}[/cyan] "
                f"first drifted at [yellow]{step_label}[/yellow] "
                f"(incr L2 = {alert.first_drift_incr_l2:.3e}, z = {alert.modified_zscore:.1f})"
            )
        if len(alerts) > 5:
            console.print(f"  [dim]… and {len(alerts) - 5} more[/dim]")
    else:
        console.print("[green]No divergent layers detected.[/green]")

    # --- Per-layer trend table ---
    top_layers = top_layers_by_drift(layer_series, metric=metric, top=top)

    table = Table(
        show_header=True,
        header_style="bold",
        title=f"Top {top} layers by {metric}  ({', '.join(labels)})",
        title_style="bold cyan",
    )
    table.add_column("Layer", style="cyan", no_wrap=True)
    table.add_column("Shape", justify="right")
    for label in labels:
        table.add_column(label, justify="right", min_width=7)
    table.add_column("trend", ratio=1)

    for name, final_value, l2_vals in top_layers:
        series = layer_series[name]
        row_style = "bold red" if any(a.layer_name == name for a in alerts) else None
        snapshot_vals = [s.l2_norm for s in series.snapshots]
        trend = _draw_trend_line(snapshot_vals, width=20)
        row = [_truncate(name, 40), "x".join(str(d) for d in series.shape)]
        for val in snapshot_vals:
            row.append(f"{val:.2e}" if np.isfinite(val) else "∞")
        row.append(trend)
        table.add_row(*row, style=row_style)

    console.print(table)

    if checkpoints[0].path.parent.name:
        console.print(f"[dim]Checkpoint dir: {checkpoints[0].path.parent}[/dim]")


def render_audit(
    report,  # AuditReport
    *,
    console: Console | None = None,
    top_outliers: int = 10,
) -> None:
    """Render an AuditReport to the terminal."""
    from safediff.audit import AuditReport

    console = console or Console()
    assert isinstance(report, AuditReport)

    path_str = str(report.path) if report.path != Path("<unknown>") else "<stdin>"

    # Header
    if report.is_healthy:
        status_text = "[bold green]OK healthy[/bold green]"
    else:
        status_text = "[bold red]ISSUES FOUND[/bold red]"
    console.print(
        f"[bold]safediff audit[/bold]  {report.total_layers} layers, "
        f"{report.total_params:,} params  {status_text}"
    )
    console.print(f"[dim]File: {path_str}[/dim]")

    # Critical: NaN
    if report.nan_layers:
        console.print(f"\n[bold red]CRITICAL:  {len(report.nan_layers)} layer(s) contain NaN[/bold red]")
        tbl = Table(
            show_header=True, header_style="bold", title="NaN layers", title_style="bold red"
        )
        tbl.add_column("Layer", style="red")
        tbl.add_column("Shape", justify="right")
        tbl.add_column("min", justify="right")
        tbl.add_column("max", justify="right")
        tbl.add_column("mean", justify="right")
        for r in report.nan_layers[:top_outliers]:
            tbl.add_row(
                _truncate(r.name, 50),
                "x".join(str(d) for d in r.shape),
                f"{r.min_val:.3e}",
                f"{r.max_val:.3e}",
                f"{r.mean_val:.3e}",
            )
        console.print(tbl)

    # Critical: Inf
    if report.inf_layers:
        console.print(f"\n[bold red]CRITICAL:  {len(report.inf_layers)} layer(s) contain Inf[/bold red]")
        tbl = Table(
            show_header=True, header_style="bold", title="Inf layers", title_style="bold red"
        )
        tbl.add_column("Layer", style="red")
        tbl.add_column("Shape", justify="right")
        tbl.add_column("min", justify="right")
        tbl.add_column("max", justify="right")
        for r in report.inf_layers[:top_outliers]:
            inf_type = []
            if r.has_pos_inf:
                inf_type.append("+Inf")
            if r.has_neg_inf:
                inf_type.append("-Inf")
            tbl.add_row(
                _truncate(r.name, 50),
                "x".join(str(d) for d in r.shape),
                f"{r.min_val:.3e}" + (" (+Inf)" if r.has_pos_inf else ""),
                f"{r.max_val:.3e}" + (" (-Inf)" if r.has_neg_inf else ""),
            )
        console.print(tbl)

    # Outliers
    if report.outlier_layers:
        console.print(
            f"\n[bold yellow]WARNING:  {len(report.outlier_layers)} layer(s) with extreme outliers "
            f"(>{r.outlier_sigma}σ from mean)[/bold yellow]"
        )
        tbl = Table(
            show_header=True,
            header_style="bold",
            title=f"Outlier layers (> {report.outlier_layers[0].outlier_sigma}σ)",
            title_style="bold yellow",
        )
        tbl.add_column("Layer", style="yellow")
        tbl.add_column("Shape", justify="right")
        tbl.add_column("outliers / total", justify="right")
        tbl.add_column("fraction", justify="right")
        tbl.add_column("range", justify="right")
        tbl.add_column("dist", ratio=1)
        peak = report.outlier_layers[0].outlier_fraction
        for r in report.outlier_layers[:top_outliers]:
            bar = _bar(r.outlier_fraction, peak, width=15)
            tbl.add_row(
                _truncate(r.name, 45),
                "x".join(str(d) for d in r.shape),
                f"{r.outlier_count:,} / {r.numel:,}",
                f"{r.outlier_fraction * 100:.3f}%",
                f"[{r.mean_val - r.outlier_sigma * r.std_val:.1e}, {r.mean_val + r.outlier_sigma * r.std_val:.1e}]",
                bar,
            )
        console.print(tbl)

    # Near-zero
    if report.near_zero_layers:
        console.print(
            f"\n[bold yellow]WARNING:  {len(report.near_zero_layers)} layer(s) are near-zero "
            f"(>90% of weights ≈ 0)[/bold yellow]"
        )
        tbl = Table(
            show_header=True,
            header_style="bold",
            title="Near-zero layers",
            title_style="bold yellow",
        )
        tbl.add_column("Layer", style="yellow")
        tbl.add_column("Shape", justify="right")
        tbl.add_column("near-zero / total", justify="right")
        tbl.add_column("fraction", justify="right")
        tbl.add_column("distribution", ratio=1)
        peak = report.near_zero_layers[0].near_zero_fraction
        for r in report.near_zero_layers[:top_outliers]:
            bar = _bar(r.near_zero_fraction, peak, width=20)
            tbl.add_row(
                _truncate(r.name, 50),
                "x".join(str(d) for d in r.shape),
                f"{int(r.near_zero_fraction * r.numel):,} / {r.numel:,}",
                f"{r.near_zero_fraction * 100:.2f}%",
                bar,
            )
        console.print(tbl)

    # Frozen layers
    if report.frozen_layers:
        console.print(
            f"\n[bold yellow]WARNING:  {len(report.frozen_layers)} frozen layer pair(s) "
            f"(nearly identical weights)[/bold yellow]"
        )
        for a, b in report.frozen_layers[:10]:
            console.print(f"  [dim]≈[/dim]  [cyan]{a}[/cyan]  ≈  [cyan]{b}[/cyan]")

    if report.is_healthy:
        console.print(
            "\n[bold green]OK:  No numerical issues detected. Model looks healthy.[/bold green]"
        )


# ------------------------------------------------------------------------------------------
# Quantisation pre-flight report rendering
# ------------------------------------------------------------------------------------------

from safediff.quant import QuantLayerStat, QuantReport


def _quant_health_color(score: float) -> str:
    if score >= 80:
        return "green"
    if score >= 50:
        return "yellow"
    return "red"


def _scheme_color(scheme: str) -> str:
    if scheme == "per-tensor":
        return "green"
    if scheme == "per-channel":
        return "yellow"
    return "red"


def render_quant_report(
    report: QuantReport,
    *,
    console: Console | None = None,
    top: int = 15,
    bits: int = 4,
) -> None:
    """Render a QuantReport to the terminal.

    Args:
        report: The output of ``quant.analyze()``.
        console: Rich console to write to.  Defaults to stdout.
        top: Show only the top-N most dangerous layers.
        bits: Which bit width to display recommendations for.
    """
    console = console or Console()

    # --- Header ---
    score_color = _quant_health_color(report.overall_score)
    console.print(
        f"[bold]safediff quant[/bold]  {report.total_layers} layers, "
        f"{report.total_params:,} params  "
        f"[bold {score_color}]score={report.overall_score:.1f}[/bold {score_color}]"
    )
    console.print(f"[dim]File: {report.path}[/dim]")

    # --- Summary row ---
    summary_parts = []
    if report.danger_count:
        summary_parts.append(f"[bold red]⛔ {report.danger_count} danger[/bold red]")
    if report.warning_count:
        summary_parts.append(f"[bold yellow]⚠  {report.warning_count} caution[/bold yellow]")
    if report.healthy_count:
        summary_parts.append(f"[bold green]✅ {report.healthy_count} healthy[/bold green]")
    if report.skip_count:
        summary_parts.append(f"[dim]⊘ {report.skip_count} skip[/dim]")
    if report.per_channel_count:
        summary_parts.append(f"[yellow]⟲ {report.per_channel_count} per-channel[/yellow]")
    console.print("  ".join(summary_parts) if summary_parts else "[green]all layers clean[/green]")

    if report.worst_offender:
        console.print(f"[dim]Worst offender: {report.worst_offender}[/dim]")

    if not report.layers:
        console.print("[yellow]No layers to display.[/yellow]")
        return

    # --- Per-layer table ---
    table = Table(
        show_header=True,
        header_style="bold",
        title=f"Top {min(top, len(report.layers))} layers by quantisation health (sorted worst → best)",
        title_style="bold cyan",
    )
    table.add_column("Layer", style="cyan", no_wrap=True)
    table.add_column("Shape", justify="right")
    table.add_column("Bits", justify="right")
    table.add_column("Scheme", justify="center")
    table.add_column("Clip%", justify="right")
    table.add_column("Outlier%", justify="right")
    table.add_column("Rel.MSE", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Distribution", ratio=1)

    def _fmt_frac(f: float) -> str:
        return f"{f * 100:.3f}%"

    display_layers = report.layers[:top]
    peak_score = max(s.health_score for s in report.layers) if report.layers else 100.0

    for stat in display_layers:
        scheme = stat.recommended
        if scheme is None:
            continue

        health_color = _quant_health_color(stat.health_score)
        scheme_color = _scheme_color(scheme.suggested_scheme)
        row_style = f"bold {health_color}" if stat.health_score < 50 else None
        score_bar = _bar(stat.health_score, peak_score, width=10)

        table.add_row(
            _truncate(stat.name, 40),
            "x".join(str(d) for d in stat.shape),
            str(scheme.bits),
            f"[{scheme_color}]{scheme.suggested_scheme}[/{scheme_color}]",
            _fmt_frac(scheme.clip_ratio),
            _fmt_frac(scheme.outlier_ratio),
            f"{scheme.error_estimate:.4f}",
            f"[{health_color}]{stat.health_score:.0f} {score_bar}[/{health_color}]",
            "",
            style=row_style,
        )

    console.print(table)

    # --- Hints section ---
    skip_layers = [s for s in report.layers if s.recommended and s.recommended.suggested_scheme == "skip"]
    if skip_layers:
        console.print(
            f"\n[bold yellow]⊘ {len(skip_layers)} layer(s) recommended to skip at {bits}bit[/bold yellow]"
        )
        for s in skip_layers[:5]:
            console.print(f"  [dim]≈[/dim]  [cyan]{s.name}[/cyan]")
        if len(skip_layers) > 5:
            console.print(f"  [dim]… and {len(skip_layers) - 5} more[/dim]")

    per_channel_layers = [
        s for s in report.layers
        if s.recommended and s.recommended.suggested_scheme == "per-channel"
    ]
    if per_channel_layers:
        console.print(
            f"\n[bold yellow]⟲ {len(per_channel_layers)} layer(s) may need per-channel quantization[/bold yellow]"
        )
        for s in per_channel_layers[:5]:
            console.print(f"  [dim]⟲[/dim]  [cyan]{s.name}[/cyan]")
        if len(per_channel_layers) > 5:
            console.print(f"  [dim]… and {len(per_channel_layers) - 5} more[/dim]")
