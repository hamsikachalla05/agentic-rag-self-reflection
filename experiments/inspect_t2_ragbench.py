#!/usr/bin/env python3
"""Print T²-RAGBench schemas and split sizes. Does not run retrieval."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loader.t2_ragbench_loader import inspect_all_configs


def main() -> None:
    report = inspect_all_configs()
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
