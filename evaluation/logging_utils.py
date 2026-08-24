"""Shared experiment logging. One JSON object per question, JSONL file plus per-question files."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import CONFIG, Config


class ExperimentLogger:
    def __init__(self, run_name: str, cfg: Config | None = None):
        self.cfg = cfg or CONFIG
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_dir = self.cfg.logs_dir / f"{run_name}_{stamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.run_dir / "records.jsonl"

    def write(self, record: dict[str, Any]) -> Path:
        path = self.run_dir / f"{record['question_id']}.json"
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return path
