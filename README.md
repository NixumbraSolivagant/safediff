# safediff

> **LLM checkpoint analysis SDK** — pure numpy core, built for quantisation engineers.

Zero-dependency analysis core (`numpy` + `safetensors`); optional PyTorch for `.pt` / `.pth` / `.bin` files. No GPU, no training loop, no web UI.

---

## Two entry points

### Python SDK (recommended)

```python
import safediff as sd

# Quant pre-flight: scan before running AWQ / GPTQ / vLLM
report = sd.quant_analyze("model.safetensors")
for layer in report.layers:
    print(f"{layer.name}: scheme={layer.recommended.suggested_scheme}, score={layer.health_score:.0f}")

# Training loop integration: zero IO, in-memory drift detection
tracker = sd.Tracker(
    baseline_state_dict={k: v.cpu().numpy() for k, v in model.state_dict().items()},
    step_id=0,
    logger=wandb.run,      # optional
    anomaly_threshold=3.5,
)
for step in range(1, 10001):
    model.train_step()
    alerts = tracker.update(
        {k: v.cpu().numpy() for k, v in model.state_dict().items()},
        step_id=step,
    )
    if alerts:
        print(f"Drift in {alerts[0].layer_name} at step {step}")

# Two-checkpoint comparison
diff = sd.analyze(checkpoint_a, checkpoint_b)
```

### CLI

```bash
pip install safediff

safediff quant   model.safetensors   # quant pre-flight (new)
safediff track  ./checkpoints/       # trace weight drift over time
safediff audit  model.safetensors   # static health scan
safediff compare A.safetensors B.safetensors  # A/B diff
```

---

## Install

```bash
# Core — .safetensors only (recommended; ~5 MB)
pip install safediff

# Add .pt / .pth / .bin support
pip install 'safediff[torch]'

# Built-in demo (downloads tiny GPT-2 on first run)
pip install 'safediff[demo]'
```

---

## `safediff quant` — Quant pre-flight scan

Before running a slow quantisation job, scan the checkpoint to know which layers need attention:

```bash
$ safediff quant model.safetensors

safediff quant  48 layers, 124,432,128 params  score=87.3
File: model.safetensors
  ⛔ 2 danger  ⚠  3 caution  ✅ 43 healthy  ⊘ 2 skip  ⟲ 1 per-channel

Top 15 layers by quantisation health (sorted worst → best)
┏━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━┓
┃ Layer        ┃ Shape  ┃ Bits  ┃ Scheme      ┃ Clip%  ┃ Outlier%┃ Rel.MSE ┃ Score ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━┩
│ lm_head…     │ 32000× │     4 │ per-tensor  │  0.4%  │  0.3%   │  0.0012 │   82  │
│ layer.11…    │  4096× │     4 │ per-tensor  │  0.2%  │  0.5%   │  0.0008 │   88  │
│ layer.15…    │  4096× │     4 │ skip        │  6.1%  │  8.2%   │  0.0231 │   41  │

⊘ 2 layer(s) recommended to skip at 4bit
  ≈  layer.15.mlp.down_proj.weight
  ≈  layer.11.attn.out_proj.weight
```

### How scheme selection works

For each tensor, `suggest_scheme()` evaluates both `per-tensor` and `per-channel`:

| Condition | Suggested scheme | Reason |
|---|---|---|
| outlier_ratio < 0.1%, clip_ratio < 0.1% | `per-tensor` | Clean distribution |
| outlier_ratio < 1% | `per-tensor` | Acceptable; warn in reason |
| outlier_ratio > 5% | `skip` | Quantisation error too high even at per-channel |
| moderate outliers, 2D tensor | `per-channel` | Falls back if it reduces outlier_ratio < 1% |

Outlier detection uses **MAD-based modified Z-score** (resistant to the masking effect where extreme outliers inflate std and hide each other).

---

## Training loop integration

Drop `sd.Tracker` into any training loop. No disk IO — it receives numpy arrays directly:

```python
import safediff as sd
import wandb

tracker = sd.Tracker(
    baseline_state_dict={k: v.cpu().numpy() for k, v in model.state_dict().items()},
    step_id=0,
    logger=wandb.run,
    log_every=100,
    anomaly_threshold=3.5,
)

for step in range(1, 50001):
    loss = model(batch)
    loss.backward()
    optimizer.step()

    alerts = tracker.update(
        {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()},
        step_id=step,
    )
    if alerts:
        # Which layer started drifting and when?
        for a in alerts:
            print(f"Drift detected: {a.layer_name} at step {step}, z={a.modified_zscore:.1f}")
```

Per-layer `drift/incr_l2/{layer}` metrics are pushed to your logger at `log_every` intervals. Works with wandb, tensorboard, or any callable.

---

## `safediff track` — Checkpoint directory analysis

Trace per-layer weight evolution across a sequence of saves:

```bash
safediff track ./checkpoints --top 8
```

Identifies which layer first started drifting, even before your loss curve spiked.

---

## `safediff audit` — Static health scan

Quick sanity check before loading into GPU:

```bash
safediff audit model.safetensors
```

Scans for extreme outliers (blocks quantisation), near-zero layers (wasted memory), NaN/Inf (hint-level — PyTorch catches these at runtime), and frozen layer pairs (copy-paste bugs).

---

## Architecture

```
safediff/
├── quant.py         # Quant pre-flight SDK (analyze, suggest_scheme)
├── integrations.py   # Tracker class (training loop integration)
├── audit.py         # Static health checks
├── track.py         # Learning dynamics tracker
├── analyzer.py      # Two-checkpoint diff engine
├── loader.py        # Format-agnostic tensor loading
└── visualizer.py    # Rich terminal rendering

tests/
├── test_quant.py        # 18 test cases
├── test_integrations.py # 14 test cases
├── test_audit.py        # 27 test cases
└── ...
```

All core analysis is **pure numpy** — no PyTorch dependency at runtime.

## Public API summary

```python
# Quant pre-flight
sd.quant_analyze(path)          # -> QuantReport
sd.suggest_scheme(arr, name)   # -> dict[int, QuantScheme]

# Training loop
sd.Tracker(...)                 # embeddable tracker class
sd.compute_delta(...)          # core delta engine
sd.anomaly_score(...)          # MAD-based z-score

# Classic tools
sd.analyze(a, b)               # two-checkpoint diff
sd.audit(tensors)             # static health scan
sd.track(checkpoints, loader)  # checkpoint directory tracking
```

## License

MIT
