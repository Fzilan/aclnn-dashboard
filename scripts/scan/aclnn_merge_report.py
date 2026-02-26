#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable


def _read_csv_map(path: Path, key: str) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        out: dict[str, dict[str, str]] = {}
        for row in r:
            k = row.get(key, "").strip()
            if not k:
                continue
            out[k] = row
        return out


def _join_limit(values: Iterable[str], limit: int = 12) -> str:
    vals = [v for v in values if v]
    if not vals:
        return ""
    if len(vals) <= limit:
        return ";".join(vals)
    return ";".join(vals[:limit]) + f";...(+{len(vals) - limit})"


def _pct(num: int, den: int) -> str:
    if den == 0:
        return "0.0%"
    return f"{(num / den) * 100:.1f}%"


def write_md(path: Path, rows: list[dict[str, str]], stats: dict[str, int]) -> None:
    lines: list[str] = []
    lines.append("# ACLNN coverage comparison (torch-npu vs MindSpore)")
    lines.append("")
    total = stats["total"]
    tn_yes = stats["tn_yes"]
    ms_yes = stats["ms_yes"]
    both_yes = stats["both_yes"]
    only_tn = stats["only_tn"]
    only_ms = stats["only_ms"]
    lines.append(f"**统计（基于 {total} 个 ACLNN API）**")
    lines.append(f"- torch-npu 已接入：{tn_yes} / {total}（{_pct(tn_yes, total)}）")
    lines.append(f"- mindspore 已接入：{ms_yes} / {total}（{_pct(ms_yes, total)}）")
    lines.append(f"- 两者都接入：{both_yes} / {total}（{_pct(both_yes, total)}）")
    lines.append(f"- 仅 torch-npu：{only_tn} / {total}（{_pct(only_tn, total)}）")
    lines.append(f"- 仅 mindspore：{only_ms} / {total}（{_pct(only_ms, total)}）")
    lines.append("")
    lines.append("计算公式：`占比 = 对应数量 / ACLNN 总数`")
    lines.append(
        "复算命令：`python3 scripts/scan/aclnn_merge_report.py --torch-npu-csv data/reports/aclnn_to_torch_npu.csv --mindspore-csv data/reports/aclnn_to_mindspore.csv --out-md data/reports/aclnn_to_all.md --out-csv data/reports/aclnn_to_all.csv`"
    )
    lines.append("")
    lines.append("| ACLNN API | torch-npu | via (ATen ops) | mindspore | pyboost | kbk | via (MS ops) |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        line = "| {aclnn_api} | {tn_status} | {tn_via} | {ms_status} | {ms_pyboost} | {ms_kbk} | {ms_via} |".format(
            **r
        )
        lines.append(line)
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "aclnn_api",
        "torch_npu",
        "torch_npu_via_ops",
        "mindspore",
        "mindspore_pyboost",
        "mindspore_kbk",
        "mindspore_via_ops",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "aclnn_api": r["aclnn_api"],
                    "torch_npu": r["tn_status"],
                    "torch_npu_via_ops": r["tn_via"],
                    "mindspore": r["ms_status"],
                    "mindspore_pyboost": r["ms_pyboost"],
                    "mindspore_kbk": r["ms_kbk"],
                    "mindspore_via_ops": r["ms_via"],
                }
            )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Merge ACLNN coverage tables for torch-npu and MindSpore.")
    ap.add_argument("--torch-npu-csv", type=Path, default=Path("data/reports/aclnn_to_torch_npu.csv"))
    ap.add_argument("--mindspore-csv", type=Path, default=Path("data/reports/aclnn_to_mindspore.csv"))
    ap.add_argument("--out-md", type=Path, default=Path("data/reports/aclnn_to_all.md"))
    ap.add_argument("--out-csv", type=Path, default=Path("data/reports/aclnn_to_all.csv"))
    args = ap.parse_args(argv)

    tn = _read_csv_map(args.torch_npu_csv, "aclnn_api")
    ms = _read_csv_map(args.mindspore_csv, "aclnn_api")

    keys = sorted(set(tn.keys()) | set(ms.keys()))
    rows: list[dict[str, str]] = []
    for k in keys:
        tn_row = tn.get(k, {})
        ms_row = ms.get(k, {})

        tn_status = "✅" if tn_row.get("torch_npu_status") == "已接入" else "✖️"
        ms_status = ms_row.get("mindspore", "✖️")

        rows.append(
            {
                "aclnn_api": k,
                "tn_status": tn_status,
                "tn_via": tn_row.get("via_op_names", ""),
                "ms_status": ms_status,
                "ms_pyboost": ms_row.get("pyboost", "✖️"),
                "ms_kbk": ms_row.get("kbk", "✖️"),
                "ms_via": ms_row.get("via_ops", ""),
            }
        )

    total = len(rows)
    tn_yes = sum(1 for r in rows if r["tn_status"] == "✅")
    ms_yes = sum(1 for r in rows if r["ms_status"] == "✅")
    both_yes = sum(1 for r in rows if r["tn_status"] == "✅" and r["ms_status"] == "✅")
    only_tn = sum(1 for r in rows if r["tn_status"] == "✅" and r["ms_status"] == "✖️")
    only_ms = sum(1 for r in rows if r["tn_status"] == "✖️" and r["ms_status"] == "✅")
    stats = {
        "total": total,
        "tn_yes": tn_yes,
        "ms_yes": ms_yes,
        "both_yes": both_yes,
        "only_tn": only_tn,
        "only_ms": only_ms,
    }

    write_md(args.out_md, rows, stats)
    write_csv(args.out_csv, rows)

    print(f"total_aclnn={total}")
    print(f"torch_npu_supported={tn_yes}")
    print(f"mindspore_supported={ms_yes}")
    print(f"both_supported={both_yes}")
    print(f"only_torch_npu={only_tn}")
    print(f"only_mindspore={only_ms}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
