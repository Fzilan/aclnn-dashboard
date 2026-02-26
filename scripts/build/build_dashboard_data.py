#!/usr/bin/env python3
"""Build dashboard-friendly JSON from ACLNN coverage reports."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    dashboard_dir = root
    parser = argparse.ArgumentParser(description="Build ACLNN dashboard data.json")
    parser.add_argument("--aclnn-md", default=str(root / "data" / "raw" / "aclnn-aa.generated.md"))
    parser.add_argument("--torch-csv", default=str(root / "data" / "reports" / "aclnn_to_torch_npu.csv"))
    parser.add_argument("--torch-md", default=str(root / "data" / "reports" / "aclnn_to_torch_npu.md"))
    parser.add_argument("--mindspore-csv", default=str(root / "data" / "reports" / "aclnn_to_mindspore.csv"))
    parser.add_argument("--mindspore-md", default=str(root / "data" / "reports" / "aclnn_to_mindspore.md"))
    parser.add_argument("--merged-csv", default=str(root / "data" / "reports" / "aclnn_to_all.csv"))
    parser.add_argument("--merged-md", default=str(root / "data" / "reports" / "aclnn_to_all.md"))
    parser.add_argument("--history-file", default=str(dashboard_dir / "coverage_history.json"))
    parser.add_argument("--output", default=str(dashboard_dir / "data.json"))
    return parser.parse_args()


def read_csv(path: Path, key: str) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            api = (row.get(key) or "").strip()
            if api:
                rows[api] = {k: (v or "").strip() for k, v in row.items()}
    return rows


def read_md_table_apis(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    table_lines = [line for line in lines if line.startswith("|")]
    if len(table_lines) < 3:
        return []

    header = [c.strip().lower() for c in table_lines[0].strip("|").split("|")]
    aclnn_idx = None
    for idx, col in enumerate(header):
        if "aclnn" in col:
            aclnn_idx = idx
            break
    if aclnn_idx is None:
        return []

    apis: list[str] = []
    for line in table_lines[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        if aclnn_idx < len(cells):
            api = cells[aclnn_idx]
            if api.startswith("aclnn"):
                apis.append(api)
    return apis


def as_list(raw: str) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(";") if item.strip()]


def flag(raw: str) -> bool:
    normalized = raw.strip().lower()
    return normalized in {"✅", "已接入", "true", "yes", "1"}


def iso_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")


def load_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        date = item.get("date")
        if not isinstance(date, str) or not date:
            continue
        out.append(item)
    out.sort(key=lambda x: x.get("date", ""))
    return out


def speed_pp_per_day(history: list[dict[str, Any]], key: str, days: int = 7) -> float:
    tail = history[-days:]
    if len(tail) < 2:
        return 0.0
    first = float(tail[0].get(key, 0.0))
    last = float(tail[-1].get(key, 0.0))
    span = max(1, len(tail) - 1)
    return round((last - first) / span, 3)


def build_data(args: argparse.Namespace) -> dict[str, Any]:
    aclnn_md = Path(args.aclnn_md)
    torch_csv = Path(args.torch_csv)
    torch_md = Path(args.torch_md)
    ms_csv = Path(args.mindspore_csv)
    ms_md = Path(args.mindspore_md)
    merged_csv = Path(args.merged_csv)
    merged_md = Path(args.merged_md)
    history_file = Path(args.history_file)
    output = Path(args.output)

    source_files = [aclnn_md, torch_csv, torch_md, ms_csv, ms_md, merged_csv, merged_md]
    for path in source_files:
        if not path.exists():
            raise FileNotFoundError(f"Required source not found: {path}")

    merged_rows = read_csv(merged_csv, key="aclnn_api")
    torch_rows = read_csv(torch_csv, key="aclnn_api")
    ms_rows = read_csv(ms_csv, key="aclnn_api")

    md_counts = {
        "aclnn_md_rows": len(read_md_table_apis(aclnn_md)),
        "torch_md_rows": len(read_md_table_apis(torch_md)),
        "mindspore_md_rows": len(read_md_table_apis(ms_md)),
        "merged_md_rows": len(read_md_table_apis(merged_md)),
    }

    operators: list[dict[str, Any]] = []
    all_apis = sorted(set(merged_rows) | set(torch_rows) | set(ms_rows))
    for api in all_apis:
        merged = merged_rows.get(api, {})
        torch = torch_rows.get(api, {})
        ms = ms_rows.get(api, {})

        torch_supported = flag(merged.get("torch_npu", "")) or flag(torch.get("torch_npu_status", ""))
        ms_supported = flag(merged.get("mindspore", "")) or flag(ms.get("mindspore", ""))
        coverage_status = (
            "both"
            if torch_supported and ms_supported
            else "torch_only"
            if torch_supported
            else "mindspore_only"
            if ms_supported
            else "neither"
        )

        operators.append(
            {
                "aclnn_api": api,
                "tag": torch.get("tag", "all") or "all",
                "coverage_status": coverage_status,
                "torch": {
                    "supported": torch_supported,
                    "status": "已接入" if torch_supported else "未接入",
                    "via_ops": as_list(torch.get("via_op_names", merged.get("torch_npu_via_ops", ""))),
                    "direct_match_ops": as_list(torch.get("direct_match_ops", "")),
                    "evidence": as_list(torch.get("evidence", "")),
                    "suspected_fusion": torch.get("suspected_fusion", "").strip().lower() == "true",
                    "cpp_funcs": as_list(torch.get("via_cpp_funcs", "")),
                    "cpp_files": as_list(torch.get("cpp_files", "")),
                    "remarks": torch.get("remarks", ""),
                },
                "mindspore": {
                    "supported": ms_supported,
                    "status": "已接入" if ms_supported else "未接入",
                    "pyboost": flag(merged.get("mindspore_pyboost", ms.get("pyboost", ""))),
                    "kbk": flag(merged.get("mindspore_kbk", ms.get("kbk", ""))),
                    "via_ops": as_list(ms.get("via_ops", merged.get("mindspore_via_ops", ""))),
                    "direct_match_ops": as_list(ms.get("direct_match", "")),
                    "evidence": as_list(ms.get("evidence", "")),
                    "suspected_fusion": ms.get("suspected_fusion", "").strip().lower() == "true",
                    "cpp_funcs": as_list(ms.get("cpp_funcs", "")),
                    "kernel_mod_files": as_list(ms.get("kernel_mod_files", "")),
                    "pyboost_files": as_list(ms.get("pyboost_files", "")),
                    "custom_dispatch_names": as_list(ms.get("custom_dispatch_names", "")),
                    "remarks": ms.get("remarks", ""),
                },
            }
        )

    total_ops = len(operators)
    torch_supported = sum(1 for op in operators if op["torch"]["supported"])
    ms_supported = sum(1 for op in operators if op["mindspore"]["supported"])
    both_supported = sum(1 for op in operators if op["coverage_status"] == "both")
    torch_only = sum(1 for op in operators if op["coverage_status"] == "torch_only")
    ms_only = sum(1 for op in operators if op["coverage_status"] == "mindspore_only")
    unsupported = sum(1 for op in operators if op["coverage_status"] == "neither")

    latest_update = max(iso_mtime(path) for path in source_files)
    history = load_history(history_file)
    data = {
        "metrics": {
            "total_ops": total_ops,
            "torch_supported": torch_supported,
            "mindspore_supported": ms_supported,
            "both_supported": both_supported,
            "torch_only": torch_only,
            "mindspore_only": ms_only,
            "unsupported": unsupported,
            "torch_coverage_rate": round((torch_supported / total_ops) * 100, 1) if total_ops else 0.0,
            "mindspore_coverage_rate": round((ms_supported / total_ops) * 100, 1) if total_ops else 0.0,
            "both_coverage_rate": round((both_supported / total_ops) * 100, 1) if total_ops else 0.0,
            "last_update_time": latest_update,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "torch_speed_pp_per_day_7d": speed_pp_per_day(history, "torch_coverage_rate", days=7),
            "mindspore_speed_pp_per_day_7d": speed_pp_per_day(history, "mindspore_coverage_rate", days=7),
        },
        "meta": {
            "source_files": [
                {
                    "path": str(path),
                    "mtime": iso_mtime(path),
                }
                for path in source_files
            ],
            "markdown_row_counts": md_counts,
        },
        "history": {
            "daily_coverage": history,
            "last_7_days": history[-7:],
        },
        "operators": operators,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def main() -> None:
    args = parse_args()
    data = build_data(args)
    metrics = data["metrics"]
    print(
        "Built data.json: "
        f"total={metrics['total_ops']}, torch={metrics['torch_supported']}, "
        f"mindspore={metrics['mindspore_supported']}, both={metrics['both_supported']}"
    )


if __name__ == "__main__":
    main()
