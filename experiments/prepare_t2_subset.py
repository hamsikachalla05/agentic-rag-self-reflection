#!/usr/bin/env python3
"""Download a 15-example T²-RAGBench PDF subset (5 per configuration)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loader.t2_ragbench_loader import prepare_small_subset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-config", type=int, default=5)
    args = parser.parse_args()
    summary = prepare_small_subset(n_per_config=args.n_per_config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
