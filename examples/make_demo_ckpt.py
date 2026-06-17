"""Generate a pair of demo checkpoints for offline experimentation.

Run:
    python examples/make_demo_ckpt.py /tmp/demo

It writes ``demo_a.safetensors`` and ``demo_b.safetensors`` to the directory
you pass on the command line. The two files are crafted so that:

* one layer in B has its weights inflated 1000x (anomaly demo)
* one layer in B has 30% of its weights held constant (dead-neuron demo)
"""

from __future__ import annotations

import sys
from pathlib import Path

from safediff.demo import demo_to_disk


def main() -> None:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "./demo_checkpoints")
    a, b = demo_to_disk(out_dir)
    print(f"wrote: {a}\nwrote: {b}")


if __name__ == "__main__":
    main()
