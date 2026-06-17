# safediff

> **Stop guessing why your model collapsed. Diff your weights like you diff your code.**

`pip install safediff` and you'll be diffing model checkpoints in 30 seconds.

---

## Why?

You're training a model. You save checkpoints at every epoch. Suddenly at epoch 47, the loss spikes to NaN. You open TensorBoard, stare at the loss curve, and... have no idea which layer exploded.

`safediff` is a tiny zero-dependency CLI that compares two checkpoints (`.safetensors` / `.pt` / `.pth` / `.bin`) and tells you:

- which layers changed the most
- which layers are statistical anomalies (probably the one that exploded)
- which parameters are "dead" (stuck at zero — wasted compute)
- a sparkline of the change distribution per layer, right in your terminal

No web UI. No account. No telemetry. One Python script's worth of code.

## Install

```bash
# For .safetensors only (recommended; ~5MB install)
pip install safediff

# Add support for .pt / .pth / .bin PyTorch checkpoints
pip install 'safediff[torch]'

# Add the built-in demo (downloads a tiny GPT-2 on first use)
pip install 'safediff[demo]'
```

## Quick start

```bash
# Diff two checkpoints and print a colored table
safediff diff model_epoch_10.safetensors model_epoch_47.safetensors

# Filter to attention layers only
safediff diff A.safetensors B.safetensors --filter '*.attn.*'

# Machine-readable output
safediff diff A.safetensors B.safetensors --format json --output diff.json

# Try it without any files (uses a built-in synthetic checkpoint pair)
safediff demo
```

## Sample output

```
safediff A=2,933,xxx params, B=2,933,xxx params, common=17 layers
                              Per-layer diff (sorted by L2 norm, descending)
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Layer                           ┃ Shape ┃ max|ΔW|  ┃ L2 norm  ┃ mean    ┃ std     ┃ dead ┃ ΔW distribution           ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ transformer.h.0.attn.c_attn.weight │ 32x96│ 1.000e+01│ 5.396e+02 ████████████████████ │ +5.4e+00 │ 5.8e+00 │  0.00% │  ▂▃▅▇█▇▆▅▄▃▂▁▂▃▅▇█▇▆▅▄▃▂▁▂▃▅▇█▇▆▅ │
│ transformer.h.0.mlp.c_fc.weight   │ 32x128│ 5.000e-04│ 3.162e-03 ░                     │ +0.0e+00 │ 1.6e-04 │ 30.00% │  ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁ │
│ ...                                                                                                                                │
└────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

⚠  1 anomalous layer(s) detected

                    Top 20 layers by dead-parameter ratio
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Layer                    ┃ dead   ┃ total ┃ ratio  ┃ distribution           ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ transformer.h.0.mlp.c_fc.weight │ 1,228 │ 4,096 │ 30.00% │ ████████████████████ │
└──────────────────────────────────────────────────────────────────────────────────┘
```

## How "anomaly" is decided

A layer is flagged red when its L2 norm of `ΔW` is more than `--anomaly-threshold` times the global median, **or** its max-absolute `ΔW` exceeds 1.0 (a heuristic for fp32 numerical blowup). Both numbers are tunable:

```bash
safediff diff A B --anomaly-threshold 5 --eps 1e-8
```

## How "dead" is decided

A parameter is "dead" if `|ΔW| < --eps`. Useful defaults:

- `1e-6` for fp32 training (default)
- `1e-3` for fp16 / bf16 mixed-precision training
- `1e-8` for very high-precision experiments

## Supported formats

| Format | Requires | Notes |
| --- | --- | --- |
| `.safetensors` | nothing | fast, memory-mapped, recommended |
| `.pt` / `.pth` | `pip install safediff[torch]` | full PyTorch state-dict support |
| `.bin` | `pip install safediff[torch]` | LLaMA / Falcon legacy format |

## License

MIT
