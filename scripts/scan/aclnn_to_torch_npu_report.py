#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


ACLNN_API_RE = re.compile(r"\baclnn[A-Za-z0-9_]+\b")
EXEC_NPU_CMD_RE = re.compile(r"\bEXEC_NPU_CMD(?:_[A-Z0-9_]+)?\s*\(\s*(aclnn[A-Za-z0-9_]+)\b")
DO_COMPATIBILITY_RE = re.compile(r"\bDO_COMPATIBILITY\s*\(\s*(aclnn[A-Za-z0-9_]+)\b")
CPP_FUNC_NAME_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")
CPP_EXCLUDED_FUNC_NAMES = {
    "if",
    "for",
    "while",
    "switch",
    "catch",
    "return",
    "sizeof",
    "static_assert",
}


def _split_pipe_row(line: str) -> list[str]:
    return [p.strip() for p in line.strip().strip("|").split("|")]


def _looks_like_separator_row(cells: list[str]) -> bool:
    if not cells:
        return False
    for cell in cells:
        cell = cell.strip()
        if not cell or "-" not in cell:
            return False
        if not set(cell) <= {"-", ":"}:
            return False
    return True


@dataclass(frozen=True)
class AclnnRow:
    tag: str
    api: str


def parse_aclnn_md_table(path: Path) -> list[AclnnRow]:
    """
    Parse the first markdown pipe-table in `aclnn-aa.generated.md`.
    Expected columns: Tag | ACLNN API
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    table_start = None
    header: list[str] | None = None
    for i, line in enumerate(lines):
        if not line.strip().startswith("|"):
            continue
        cells = _split_pipe_row(line)
        if len(cells) < 2:
            continue
        if i + 1 >= len(lines):
            continue
        sep = _split_pipe_row(lines[i + 1]) if lines[i + 1].strip().startswith("|") else []
        if len(sep) == len(cells) and _looks_like_separator_row(sep):
            table_start = i
            header = cells
            break
    if table_start is None or header is None:
        raise ValueError(f"No markdown table found in {path}")

    # Find indices
    try:
        tag_idx = header.index("Tag")
        api_idx = header.index("ACLNN API")
    except ValueError as exc:
        raise ValueError(f"Unexpected header in {path}: {header}") from exc

    out: list[AclnnRow] = []
    for line in lines[table_start + 2 :]:
        if not line.strip().startswith("|"):
            break
        cells = _split_pipe_row(line)
        if len(cells) != len(header):
            break
        tag = cells[tag_idx]
        api = cells[api_idx]
        if not re.match(r"^aclnn[A-Z]", api):
            continue
        out.append(AclnnRow(tag=tag, api=api))
    return out


def _func_name_from_schema(schema: str) -> str:
    schema = schema.strip()
    if "(" in schema:
        return schema.split("(", 1)[0].strip()
    return schema


def _camelize(name: str) -> str:
    parts = [p for p in re.split(r"[^A-Za-z0-9]+", name) if p]
    if not parts:
        return ""
    return "".join(p[:1].upper() + p[1:] for p in parts)


def expected_direct_aclnn(op_name: str) -> str | None:
    base = op_name.strip()
    if not base:
        return None
    base = base.split(".", 1)[0]
    if base.startswith("__") and base.endswith("__"):
        return None
    inplace = base.endswith("_")
    if inplace:
        base = base[:-1]
    base = base.lstrip("_")
    cc = _camelize(base)
    if not cc:
        return None
    return ("aclnnInplace" if inplace else "aclnn") + cc


@dataclass(frozen=True)
class OpPluginEntry:
    name: str
    section: str
    op_api_field: Any
    exec_raw: str | None
    structured_inherit: str | None


def load_op_plugin_entries(path: Path) -> dict[str, OpPluginEntry]:
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    out: dict[str, OpPluginEntry] = {}
    for section in ("official", "custom", "symint"):
        lst = obj.get(section)
        if not isinstance(lst, list):
            continue
        for it in lst:
            if not isinstance(it, dict) or not isinstance(it.get("func"), str):
                continue
            schema = it["func"]
            name = _func_name_from_schema(schema)
            gen = it.get("gen_opapi")
            exec_raw = gen.get("exec") if isinstance(gen, dict) else None
            exec_raw = exec_raw if isinstance(exec_raw, str) else None
            structured_inherit = gen.get("structured_inherit") if isinstance(gen, dict) else None
            structured_inherit = structured_inherit if isinstance(structured_inherit, str) else None
            out[name] = OpPluginEntry(
                name=name,
                section=section,
                op_api_field=it.get("op_api"),
                exec_raw=exec_raw,
                structured_inherit=structured_inherit,
            )
    return out


def _extract_exec_aclnn(exec_raw: str | None) -> str | None:
    if not exec_raw:
        return None
    m = ACLNN_API_RE.search(exec_raw)
    return m.group(0) if m else None


def resolve_op_plugin_exec(entries: dict[str, OpPluginEntry]) -> dict[str, str | None]:
    exec_map: dict[str, str | None] = {k: _extract_exec_aclnn(v.exec_raw) for k, v in entries.items()}
    for _ in range(6):
        changed = False
        for name, ent in entries.items():
            if exec_map.get(name):
                continue
            parent = ent.structured_inherit
            if parent and exec_map.get(parent):
                exec_map[name] = exec_map[parent]
                changed = True
        if not changed:
            break
    return exec_map


def op_name_to_cpp_func_candidates(op_name: str) -> list[str]:
    name = op_name.strip()
    if not name:
        return []
    candidates: set[str] = set()
    if name.endswith(".out"):
        base = name[: -len(".out")].split(".", 1)[0]
        candidates.add(f"{base}_out")
    elif name.endswith("_out"):
        base = name.split(".", 1)[0]
        candidates.add(f"{base}_out")
    base = name.split(".", 1)[0]
    candidates.add(base)
    candidates.add(name.replace(".", "_"))
    return [c for c in sorted(candidates) if c]


@dataclass(frozen=True)
class SourceScanResult:
    func_to_aclnn_calls: dict[str, set[str]]
    func_to_files: dict[str, set[str]]


def _iter_source_files(root: Path) -> Iterable[Path]:
    for ext in ("*.cpp", "*.cc", "*.cxx"):
        yield from root.rglob(ext)


def scan_opapi_sources_for_aclnn_calls(root: Path) -> SourceScanResult:
    func_to_calls: dict[str, set[str]] = {}
    func_to_files: dict[str, set[str]] = {}
    if not root.exists():
        return SourceScanResult(func_to_aclnn_calls={}, func_to_files={})

    for file_path in _iter_source_files(root):
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        current_func: str | None = None
        brace_depth = 0
        pending_sig = ""
        pending_name: str | None = None

        for raw_line in text.splitlines():
            line = raw_line
            if current_func is None:
                if pending_sig and ";" in line and "{" not in line:
                    pending_sig = ""
                    pending_name = None

                if "(" in line or pending_sig:
                    pending_sig = (pending_sig + " " + line.strip()).strip()
                    matches = CPP_FUNC_NAME_RE.findall(pending_sig)
                    if matches:
                        cand = matches[-1]
                        if cand not in CPP_EXCLUDED_FUNC_NAMES:
                            pending_name = cand

                if "{" in line and pending_name and "namespace" not in pending_sig:
                    current_func = pending_name
                    brace_depth = line.count("{") - line.count("}")
                    if brace_depth <= 0:
                        brace_depth = 1
                    pending_sig = ""
                    pending_name = None
                    continue

                if len(pending_sig) > 2000:
                    pending_sig = ""
                    pending_name = None
                continue

            calls = set(EXEC_NPU_CMD_RE.findall(line)) | set(DO_COMPATIBILITY_RE.findall(line))
            if calls:
                func_to_calls.setdefault(current_func, set()).update(calls)
                func_to_files.setdefault(current_func, set()).add(str(file_path))

            brace_depth += line.count("{") - line.count("}")
            if brace_depth <= 0:
                current_func = None
                brace_depth = 0

    func_to_calls = {k: v for k, v in func_to_calls.items() if v}
    func_to_files = {k: v for k, v in func_to_files.items() if k in func_to_calls}
    return SourceScanResult(func_to_aclnn_calls=func_to_calls, func_to_files=func_to_files)


def _join_limit(values: Iterable[str], limit: int = 8) -> str:
    vals = [v for v in values if v]
    if not vals:
        return ""
    if len(vals) <= limit:
        return ";".join(vals)
    return ";".join(vals[:limit]) + f";...(+{len(vals) - limit})"


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def write_md_table(path: Path, rows: list[dict[str, str]], columns: list[tuple[str, str]]) -> None:
    # columns: (header, key)
    lines: list[str] = []
    lines.append("# ACLNN -> torch-npu coverage (from ACLNN list)")
    lines.append("")
    lines.append("| " + " | ".join(h for h, _ in columns) + " |")
    lines.append("|" + "|".join("---" for _ in columns) + "|")
    for r in rows:
        vals = []
        for _, key in columns:
            v = (r.get(key) or "").replace("\n", " ").strip()
            # avoid breaking markdown table
            v = v.replace("|", "\\|")
            vals.append(v)
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Start from ACLNN APIs and report torch-npu integration coverage.")
    ap.add_argument("--aclnn-md", type=Path, default=Path("data/raw/aclnn-aa.generated.md"))
    ap.add_argument("--op-plugin-yaml", type=Path, default=Path("op-plugin/op_plugin/config/op_plugin_functions.yaml"))
    ap.add_argument("--opapi-src-root", type=Path, default=Path("op-plugin/op_plugin/ops/opapi"))
    ap.add_argument("--out-csv", type=Path, default=Path("data/reports/aclnn_to_torch_npu.csv"))
    ap.add_argument("--out-md", type=Path, default=Path("data/reports/aclnn_to_torch_npu.md"))
    args = ap.parse_args(argv)

    aclnn_rows = parse_aclnn_md_table(args.aclnn_md)
    aclnn_apis = sorted({r.api for r in aclnn_rows})
    tag_map = {r.api: r.tag for r in aclnn_rows}

    op_plugin_entries = load_op_plugin_entries(args.op_plugin_yaml)
    op_exec = resolve_op_plugin_exec(op_plugin_entries)

    # Invert YAML mapping: aclnn -> op_name(s)
    yaml_aclnn_to_ops: dict[str, set[str]] = {}
    for op_name, aclnn in op_exec.items():
        if not aclnn:
            continue
        yaml_aclnn_to_ops.setdefault(aclnn, set()).add(op_name)

    # Source scan: c++ func -> aclnn calls, then map c++ func -> op_name candidates
    scan = scan_opapi_sources_for_aclnn_calls(args.opapi_src_root)
    cand_to_ops: dict[str, set[str]] = {}
    for op_name in op_plugin_entries.keys():
        for cand in op_name_to_cpp_func_candidates(op_name):
            cand_to_ops.setdefault(cand, set()).add(op_name)

    src_aclnn_to_ops: dict[str, set[str]] = {}
    src_aclnn_to_funcs: dict[str, set[str]] = {}
    src_aclnn_to_files: dict[str, set[str]] = {}
    for func, calls in scan.func_to_aclnn_calls.items():
        ops = cand_to_ops.get(func, set())
        files = scan.func_to_files.get(func, set())
        for aclnn in calls:
            src_aclnn_to_funcs.setdefault(aclnn, set()).add(func)
            src_aclnn_to_files.setdefault(aclnn, set()).update(files)
            if ops:
                src_aclnn_to_ops.setdefault(aclnn, set()).update(ops)

    fieldnames = [
        "tag",
        "aclnn_api",
        "torch_npu_status",
        "evidence",
        "via_op_names",
        "via_op_names_yaml_exec",
        "via_op_names_src_scan",
        "via_cpp_funcs",
        "cpp_files",
        "op_api_versions",
        "direct_match_ops",
        "suspected_fusion",
        "remarks",
    ]
    out_rows: list[dict[str, str]] = []

    for aclnn in aclnn_apis:
        yaml_ops = sorted(yaml_aclnn_to_ops.get(aclnn, set()))
        src_ops = sorted(src_aclnn_to_ops.get(aclnn, set()))
        all_ops = sorted(set(yaml_ops) | set(src_ops))
        funcs = sorted(src_aclnn_to_funcs.get(aclnn, set()))
        files = sorted(src_aclnn_to_files.get(aclnn, set()))

        evidence_parts: list[str] = []
        if yaml_ops:
            evidence_parts.append("yaml_exec")
        if funcs:
            evidence_parts.append("src_scan")
        evidence = "+".join(evidence_parts) if evidence_parts else ""

        status = "未接入" if not evidence_parts else "已接入"

        # Aggregate op_api field (from op-plugin yaml) for mapped ops.
        op_api_versions: set[str] = set()
        for op_name in all_ops:
            ent = op_plugin_entries.get(op_name)
            if ent is None or ent.op_api_field is None:
                continue
            op_api_versions.add(str(ent.op_api_field))

        direct_match_ops = sorted([op for op in all_ops if expected_direct_aclnn(op) == aclnn])
        suspected_fusion = bool(status == "已接入" and not direct_match_ops)

        # Remarks: shared primitive vs direct mapping unknown at this stage
        remarks: list[str] = []
        if status == "未接入":
            remarks.append("no_yaml_exec_and_no_src_scan_hit")
        else:
            if len(all_ops) >= 2:
                remarks.append(f"shared_by_{len(all_ops)}_ops")
            if yaml_ops and not funcs:
                remarks.append("yaml_only")
            if funcs and not yaml_ops:
                remarks.append("src_only")
            if yaml_ops and funcs:
                remarks.append("yaml+src")
            if funcs and not src_ops:
                remarks.append("src_hit_but_op_name_unresolved")

        out_rows.append(
            {
                "tag": tag_map.get(aclnn, ""),
                "aclnn_api": aclnn,
                "torch_npu_status": status,
                "evidence": evidence,
                "via_op_names": _join_limit(all_ops, limit=12),
                "via_op_names_yaml_exec": _join_limit(yaml_ops, limit=12),
                "via_op_names_src_scan": _join_limit(src_ops, limit=12),
                "via_cpp_funcs": _join_limit(funcs, limit=8),
                "cpp_files": _join_limit(files, limit=6),
                "op_api_versions": _join_limit(sorted(op_api_versions), limit=6),
                "direct_match_ops": _join_limit(direct_match_ops, limit=8),
                "suspected_fusion": str(suspected_fusion),
                "remarks": ";".join(remarks),
            }
        )

    write_csv(args.out_csv, out_rows, fieldnames=fieldnames)
    write_md_table(
        args.out_md,
        out_rows,
        columns=[
            ("ACLNN API", "aclnn_api"),
            ("torch-npu", "torch_npu_status"),
            ("evidence", "evidence"),
            ("via (ATen ops)", "via_op_names"),
            ("direct match", "direct_match_ops"),
            ("suspected fusion", "suspected_fusion"),
            ("C++ funcs", "via_cpp_funcs"),
            ("remarks", "remarks"),
        ],
    )

    from collections import Counter

    c = Counter(r["torch_npu_status"] for r in out_rows)
    print(f"aclnn_total={len(out_rows)} integrated={c.get('已接入',0)} not_integrated={c.get('未接入',0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
