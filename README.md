# safediff

> **Find which layer caused the crash — before TensorBoard even noticed.**

`pip install safediff` and start auditing checkpoints in 30 seconds.

---

## What it does

`safediff` is a zero-dependency CLI for **static checkpoint analysis** — no GPU, no training loop, no web UI. Three commands:

| Command | What it does | When you need it |
|---|---|---|
| `safediff track <dir>` | Trace per-layer weight evolution across a directory of checkpoints | Your model collapsed at epoch 50 and you need to know which layer started drifting first |
| `safediff audit <file>` | Scan a single checkpoint for NaN/Inf/outliers/frozen layers | Downloading a model from HuggingFace; running quantization; pre-flight check |
| `safediff compare A B` | Per-layer diff between two checkpoints | Quick A/B comparison of two specific saves |

The killer feature of `track` is **automatic divergence detection**: it flags the first layer that started drifting, even before your loss curve spiked. That's a question TensorBoard can't answer.

## Install

```bash
# Core — .safetensors only (recommended; ~5 MB)
pip install safediff

# Add .pt / .pth / .bin support
pip install 'safediff[torch]'

# Built-in demo (downloads tiny GPT-2 on first run)
pip install 'safediff[demo]'
```

## Quick start

```bash
# 1. Track weight evolution across a checkpoint directory
safediff track ./checkpoints --top 10

# 2. Audit a single model before loading it into GPU
safediff audit model.safetensors

# 3. Compare two checkpoints (the original diff command, still available)
safediff compare epoch_10.safetensors epoch_47.safetensors

# 4. Filter to specific layers while tracking
safediff track ./checkpoints --filter "*.mlp.c_proj.*" --metric incremental_l2

# 5. Export report to file
safediff audit model.safetensors --output audit.json --format json
```

## Track — Learning Dynamics Tracker

The most powerful command. Given a directory of checkpoints, it:

1. Sorts them chronologically (by `epoch_N` / `step_N` in filename, or mtime)
2. Computes per-layer delta statistics from the baseline (checkpoint 0)
3. Computes **incremental L2** (change since the previous checkpoint)
4. Detects which layer first started diverging using **modified Z-score (MAD-based)**

```
$ safediff track ./checkpoints --top 8

safediff track  20 layers × 50 checkpoints  (epoch_01 → epoch_50)

⚠  3 layer(s) diverged during training
  ▸  transformer.h.11.mlp.c_proj.weight  first drifted at epoch_23
     (incr L2 = 2.47, z = 6.3)
  ▸  transformer.h.10.mlp.c_fc.weight   first drifted at epoch_31
     (incr L2 = 1.83, z = 4.1)
  ▸  transformer.h.9.attn.c_proj.weight  first drifted at epoch_38
     (incr L2 = 0.91, z = 3.7)

Top 8 layers by cumulative_l2  (epoch_01, epoch_10, epoch_20, epoch_30, epoch_40, epoch_50)
┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━┓
┃ Layer                  ┃  epoch_01   ┃  epoch_10  ┃  epoch_20  ┃  epoch_30  ┃  epoch_40  ┃  epoch_50  ┃ trend ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━┩
│ transformer.wte.weight │   6.21e+01  │  8.47e+01  │  1.12e+02  │  1.48e+02  │  1.89e+02  │  2.31e+02  │ ▂▄▅▆▇▇ │
│ transformer.h.11…     │   1.53e+01  │  2.87e+01  │  5.14e+01  │  8.29e+01  │  1.21e+02  │  1.68e+02  │ ▂▂▃▄▇█ │
│ transformer.h.10…     │   1.53e+01  │  2.01e+01  │  3.14e+01  │  5.82e+01  │  7.91e+01  │  1.01e+02  │ ▂▃▃▄▆▇ │
│ ...
```

The key insight: `h.11.mlp.c_proj.weight` started drifting at **epoch 23**, but the loss spike didn't appear until epoch 50. Without safediff, you'd never know which layer planted the seed.

### How divergence detection works

Uses the **Modified Z-score** based on Median Absolute Deviation (MAD), the NIST-recommended robust alternative to standard deviation:

```
z = |value - median| / (1.4826 × MAD)
```

A layer is flagged when `z > 3.5` (the NIST threshold). This is robust to the very outliers it detects — unlike using standard deviation.

## Audit — Model Sanity Checker

Run before loading a model into GPU, or before quantisation:

```bash
$ safediff audit model.safetensors

safediff audit  20 layers, 137,022,720 params  ✗ issues found
File: model.safetensors

⚠  1 layer(s) with extreme outliers (>5σ from mean)
  layer                  shape          outliers / total   fraction  range
  transformer.h.11…      3072×768       847 / 2,359,296    0.036%   [-0.72, 0.74]

✓  No NaN / Inf detected
✓  No near-zero layers
✓  No frozen layer pairs
```

Checks performed:
- **NaN / Inf** — will crash GPU kernels immediately
- **Extreme outliers** — blocks INT8 / GPTQ / AWQ quantisation
- **Near-zero layers** — >90% of weights ≈ 0 (wasted memory)
- **Frozen layer pairs** — two layers with identical weights (copy-paste bug)

## Compare — Two-way diff

The original command, available as `compare` (or `diff` for backwards compatibility):

```bash
safediff compare A.safetensors B.safetensors --filter "*.attn.*"
```

## Supported formats

| Format | Requires | Notes |
|---|---|---|
| `.safetensors` | nothing | fast, memory-mapped, recommended |
| `.pt` / `.pth` | `pip install safediff[torch]` | full PyTorch state-dict support |
| `.bin` | `pip install safediff[torch]` | LLaMA / Falcon legacy format |

## Architecture

```
safediff/
├── cli.py          # Typer CLI — 4 subcommands
├── loader.py       # Format-agnostic tensor loading (safetensors / torch)
├── analyzer.py      # Two-checkpoint diff engine (pure numpy)
├── track.py        # Learning Dynamics Tracker (pure numpy)
├── audit.py        # Model Sanity Checker (pure numpy)
├── visualizer.py   # Rich terminal rendering
└── demo.py        # Built-in demo with offline fallback
```

All core analysis is **pure numpy** — no PyTorch dependency at runtime. The torch extra is only needed to load `.pt`/`.pth`/`.bin` files.

## License

MIT
