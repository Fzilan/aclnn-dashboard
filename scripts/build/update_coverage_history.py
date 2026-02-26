#!/usr/bin/env python3
"""Update ACLNN coverage history based on dashboard data.json metrics."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Update coverage history JSON from dashboard data.json")
    parser.add_argument("--data-json", type=Path, default=base / "data.json")
    parser.add_argument("--history-file", type=Path, default=base / "coverage_history.json")
    parser.add_argument("--date", type=str, default="")
    parser.add_argument("--max-entries", type=int, default=365)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_date(raw: str) -> str:
    if raw:
        return raw
    return datetime.now(timezone.utc).date().isoformat()


def load_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        value = read_json(path)
    except Exception:
        return []
    if not isinstance(value, list):
        return []
    out = [item for item in value if isinstance(item, dict) and isinstance(item.get("date"), str)]
    out.sort(key=lambda x: x.get("date", ""))
    return out


def main() -> int:
    args = parse_args()
    payload = read_json(args.data_json)
    metrics = payload.get("metrics", {})
    if not isinstance(metrics, dict):
        raise ValueError("Invalid data.json: missing metrics object.")

    date_str = resolve_date(args.date)
    snapshot = {
        "date": date_str,
        "total_ops": int(metrics.get("total_ops", 0)),
        "torch_supported": int(metrics.get("torch_supported", 0)),
        "mindspore_supported": int(metrics.get("mindspore_supported", 0)),
        "both_supported": int(metrics.get("both_supported", 0)),
        "torch_coverage_rate": float(metrics.get("torch_coverage_rate", 0.0)),
        "mindspore_coverage_rate": float(metrics.get("mindspore_coverage_rate", 0.0)),
        "both_coverage_rate": float(metrics.get("both_coverage_rate", 0.0)),
    }

    history = load_history(args.history_file)
    by_date = {item["date"]: item for item in history}
    by_date[date_str] = snapshot
    new_history = sorted(by_date.values(), key=lambda x: x["date"])
    if args.max_entries > 0:
        new_history = new_history[-args.max_entries :]

    args.history_file.parent.mkdir(parents=True, exist_ok=True)
    args.history_file.write_text(json.dumps(new_history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Updated history: {args.history_file} entries={len(new_history)} latest={date_str}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
