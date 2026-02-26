#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


ACLNN_API_RE = re.compile(r"\baclnn[A-Za-z0-9_]+\b")
LAUNCH_ACLNN_RE = re.compile(r"\bLAUNCH_ACLNN\s*\(\s*(aclnn[A-Za-z0-9_]+)\b")

COMMON_MACRO_RE = re.compile(
    r"\bMS_ACLNN_COMMON_KERNEL_FACTORY_REG\s*\(\s*([A-Za-z0-9_]+)\s*,\s*([A-Za-z0-9_]+)\s*,"
)
KERNEL_REG_RE = re.compile(
    r"\bMS_ACLNN_KERNEL_FACTORY_REG\s*\(\s*([A-Za-z0-9_]+)\s*,\s*([A-Za-z0-9_]+)\s*\)"
)
CTOR_ACLNN_RE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)\s*\([^;{}]*\)\s*:\s*AclnnKernelMod\([^\\)]*\"(aclnn[^\"]+)\"",
    re.S,
)
WS_ACLNN_RE = re.compile(r"\bDEFINE_GET_WORKSPACE_FOR_OPS\s*\(\s*(aclnn[A-Za-z0-9_]+)\b")


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


def _snake_to_pascal(name: str) -> str:
    parts = [p for p in re.split(r"[^A-Za-z0-9]+", name) if p]
    return "".join(p[:1].upper() + p[1:] for p in parts)


def _normalize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", name).lower()


def _join_limit(values: Iterable[str], limit: int = 10) -> str:
    vals = [v for v in values if v]
    if not vals:
        return ""
    if len(vals) <= limit:
        return ";".join(vals)
    return ";".join(vals[:limit]) + f";...(+{len(vals) - limit})"


def _short_path(path: str) -> str:
    for key in ("kernel_mod_impl/", "pyboost_impl/"):
        idx = path.find(key)
        if idx >= 0:
            return path[idx:]
    return Path(path).name


def _iter_files(root: Path, exts: tuple[str, ...]) -> Iterable[Path]:
    for ext in exts:
        yield from root.rglob(f"*{ext}")


def scan_kernel_mod(root: Path):
    """Return mappings for kernel_mod_impl evidence."""
    class_to_aclnn: dict[str, set[str]] = {}
    class_to_file: dict[str, set[str]] = {}

    files = list(_iter_files(root, (".h", ".hpp", ".cc", ".cpp")))
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for m in CTOR_ACLNN_RE.finditer(text):
            cls, aclnn = m.group(1), m.group(2)
            class_to_aclnn.setdefault(cls, set()).add(aclnn)
            class_to_file.setdefault(cls, set()).add(str(path))

        # Capture workspace ops inside class definitions (rough scope tracking).
        current_class = None
        in_class = False
        brace_depth = 0
        brace_started = False
        for line in text.splitlines():
            if not in_class:
                m = re.search(r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)\b", line)
                if m:
                    current_class = m.group(1)
                    in_class = True
                    brace_depth = line.count("{") - line.count("}")
                    brace_started = brace_depth > 0
            else:
                for w in WS_ACLNN_RE.findall(line):
                    if current_class:
                        class_to_aclnn.setdefault(current_class, set()).add(w)
                        class_to_file.setdefault(current_class, set()).add(str(path))
                brace_depth += line.count("{") - line.count("}")
                if not brace_started and "{" in line:
                    brace_started = True
                if brace_started and brace_depth <= 0:
                    in_class = False
                    current_class = None
                    brace_depth = 0
                    brace_started = False

    aclnn_to_ops: dict[str, set[str]] = {}
    aclnn_to_classes: dict[str, set[str]] = {}
    aclnn_to_files: dict[str, set[str]] = {}

    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for m in COMMON_MACRO_RE.finditer(text):
            op_name, aclnn = m.group(1), m.group(2)
            if not aclnn.startswith("aclnn"):
                continue
            aclnn_to_ops.setdefault(aclnn, set()).add(op_name)
            aclnn_to_files.setdefault(aclnn, set()).add(str(path))

        for m in KERNEL_REG_RE.finditer(text):
            op_name, cls = m.group(1), m.group(2)
            if cls not in class_to_aclnn:
                continue
            for aclnn in class_to_aclnn.get(cls, set()):
                aclnn_to_ops.setdefault(aclnn, set()).add(op_name)
                aclnn_to_classes.setdefault(aclnn, set()).add(cls)
                aclnn_to_files.setdefault(aclnn, set()).add(str(path))
                aclnn_to_files.setdefault(aclnn, set()).update(class_to_file.get(cls, set()))

    return aclnn_to_ops, aclnn_to_classes, aclnn_to_files


def scan_pyboost(root: Path):
    """Return mappings for pyboost_impl evidence."""
    aclnn_to_ops: dict[str, set[str]] = {}
    aclnn_to_files: dict[str, set[str]] = {}
    aclnn_to_funcs: dict[str, set[str]] = {}

    for path in _iter_files(root, (".cc", ".cpp")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        calls = set(LAUNCH_ACLNN_RE.findall(text))
        if not calls:
            continue
        stem = path.stem
        inferred_op = _snake_to_pascal(stem)
        for aclnn in calls:
            aclnn_to_ops.setdefault(aclnn, set()).add(inferred_op)
            aclnn_to_files.setdefault(aclnn, set()).add(str(path))
            aclnn_to_funcs.setdefault(aclnn, set()).add(f"pyboost::{stem}")

    return aclnn_to_ops, aclnn_to_funcs, aclnn_to_files


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def _as_default_ascend(dispatch: dict) -> bool:
    ascend = dispatch.get("Ascend", "default")
    if ascend is None:
        return False
    if isinstance(ascend, str):
        return ascend.strip().lower() == "default"
    return False


def _load_yaml_obj(path: Path) -> dict:
    try:
        obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _iter_op_def_entries(op_def_root: Path) -> Iterable[tuple[str, dict]]:
    if not op_def_root.exists():
        return []
    entries: list[tuple[str, dict]] = []
    for y in sorted(op_def_root.glob("*_op.yaml")):
        obj = _load_yaml_obj(y)
        for op_name, op_cfg in obj.items():
            if isinstance(op_name, str) and isinstance(op_cfg, dict):
                entries.append((op_name, op_cfg))
    return entries


def scan_gen_config(op_def_root: Path, aclnn_config_yaml: Path):
    """
    Parse op_def + aclnn_config to infer potential auto-generated ACLNN mappings.
    Returns:
      gen_km_auto_ops: aclnn -> ops likely kernel_mod auto-generated path
      gen_pb_auto_ops: aclnn -> ops likely pyboost auto-generated path
      custom_dispatch_ops: aclnn -> ops with explicit Ascend customize dispatch
    """
    aclnn_map_raw = _load_yaml_obj(aclnn_config_yaml)
    aclnn_map = {str(k): str(v) for k, v in aclnn_map_raw.items() if isinstance(k, str) and isinstance(v, str)}

    gen_km_auto_ops: dict[str, set[str]] = {}
    gen_pb_auto_ops: dict[str, set[str]] = {}
    custom_dispatch_ops: dict[str, set[str]] = {}
    custom_dispatch_names: dict[str, set[str]] = {}

    for op_name, op_cfg in _iter_op_def_entries(op_def_root):
        dispatch = op_cfg.get("dispatch")
        if not isinstance(dispatch, dict) or not _to_bool(dispatch.get("enable", False)):
            continue

        class_name = op_cfg.get("class", {}).get("name") if isinstance(op_cfg.get("class"), dict) else None
        class_name = class_name if isinstance(class_name, str) and class_name else _snake_to_pascal(op_name)
        aclnn = aclnn_map.get(class_name, f"aclnn{class_name}")

        ascend = dispatch.get("Ascend", "default")
        if ascend == "None":
            continue
        if _as_default_ascend(dispatch):
            gen_km_auto_ops.setdefault(aclnn, set()).add(class_name)
            gen_pb_auto_ops.setdefault(aclnn, set()).add(class_name)
        elif isinstance(ascend, str) and ascend:
            custom_dispatch_ops.setdefault(aclnn, set()).add(class_name)
            custom_dispatch_names.setdefault(aclnn, set()).add(ascend)

    return gen_km_auto_ops, gen_pb_auto_ops, custom_dispatch_ops, custom_dispatch_names


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def write_md_table(path: Path, rows: list[dict[str, str]], columns: list[tuple[str, str]]) -> None:
    lines: list[str] = []
    lines.append("# ACLNN -> MindSpore coverage (from ACLNN list)")
    lines.append("")
    lines.append("**列名说明**")
    lines.append("- `ACLNN API`: CANN ACLNN 的算子名")
    lines.append("- `mindspore`: 是否有任一路径接入证据（✅=已接入，✖️=未接入；包含源码静态与生成配置证据）")
    lines.append("- `pyboost`: 是否有 PyBoost 路径接入证据（✅/✖️）")
    lines.append("- `kbk`: 是否有 KernelMod(KBK/Graph) 路径接入证据（✅/✖️）")
    lines.append("- `kernel_mod(src)`/`pyboost(src)`: 源码静态命中（分别扫描 `kernel_mod_impl` / `pyboost_impl`）")
    lines.append("- `kernel_mod(gen)`/`pyboost(gen)`: 生成配置推断命中（由 `op_def/yaml` + `aclnn_config.yaml` 推断默认自动生成路径）")
    lines.append("- `evidence`: 总证据来源（`kernel_mod_src`/`pyboost_src`/`kernel_mod_gen`/`pyboost_gen`）")
    lines.append("- `via (MS ops)`: 该 aclnn 对应的 MindSpore op 名（kernel_mod 注册名或 pyboost 文件名推断）")
    lines.append("- `via kernel_mod(src)` / `via pyboost(src)`: 分路径静态命中的 op")
    lines.append("- `via kernel_mod(gen)` / `via pyboost(gen)`: 分路径生成配置推断的 op")
    lines.append("- `custom dispatch ops`: `dispatch.Ascend` 显式配置的 op（通常走 customize，不等于 auto-generate）")
    lines.append("- `direct match`: 同名直连判断（大小写不敏感地比较 op 名与 aclnn 后缀，匹配到的 op 列表）")
    lines.append("- `suspected fusion`: `True` 表示有接入证据但没有同名直连匹配，可能是融合/重命名/复用更底层 primitive")
    lines.append("- `C++ funcs`: 关联的内核类名或 pyboost 文件标识（`pyboost::<file>`）")
    lines.append("- `kernel_mod files`: 来自 `kernel_mod_impl` 的命中文件（路径从 `kernel_mod_impl/` 起）")
    lines.append("- `pyboost files`: 来自 `pyboost_impl` 的命中文件（路径从 `pyboost_impl/` 起）")
    lines.append("- `remarks`: 其他备注（如来源组合、共享情况、文件线索）")
    lines.append("")
    lines.append("| " + " | ".join(h for h, _ in columns) + " |")
    lines.append("|" + "|".join("---" for _ in columns) + "|")
    for r in rows:
        vals = []
        for _, key in columns:
            v = (r.get(key) or "").replace("\n", " ").strip()
            v = v.replace("|", "\\|")
            vals.append(v)
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Start from ACLNN APIs and report MindSpore integration coverage.")
    ap.add_argument("--aclnn-md", type=Path, default=Path("data/raw/aclnn-aa.generated.md"))
    ap.add_argument(
        "--ms-root",
        type=Path,
        default=Path("mindspore/mindspore/ops/kernel/ascend/aclnn"),
    )
    ap.add_argument("--op-def-root", type=Path, default=Path("mindspore/mindspore/ops/op_def/yaml"))
    ap.add_argument(
        "--aclnn-config",
        type=Path,
        default=Path("mindspore/mindspore/python/mindspore/ops_generate/pyboost/aclnn_config.yaml"),
    )
    ap.add_argument("--out-csv", type=Path, default=Path("data/reports/aclnn_to_mindspore.csv"))
    ap.add_argument("--out-md", type=Path, default=Path("data/reports/aclnn_to_mindspore.md"))
    args = ap.parse_args(argv)

    aclnn_rows = parse_aclnn_md_table(args.aclnn_md)
    aclnn_apis = sorted({r.api for r in aclnn_rows})

    kernel_mod_root = args.ms_root / "kernel_mod_impl"
    pyboost_root = args.ms_root / "pyboost_impl"

    km_ops, km_classes, km_files = scan_kernel_mod(kernel_mod_root)
    pb_ops, pb_funcs, pb_files = scan_pyboost(pyboost_root)
    km_gen_ops, pb_gen_ops, custom_dispatch_ops, custom_dispatch_names = scan_gen_config(args.op_def_root, args.aclnn_config)

    rows: list[dict[str, str]] = []
    for aclnn in aclnn_apis:
        ops_km = sorted(km_ops.get(aclnn, set()))
        ops_pb = sorted(pb_ops.get(aclnn, set()))
        ops_km_gen = sorted(km_gen_ops.get(aclnn, set()))
        ops_pb_gen = sorted(pb_gen_ops.get(aclnn, set()))
        ops_custom_dispatch = sorted(custom_dispatch_ops.get(aclnn, set()))
        dispatch_names = sorted(custom_dispatch_names.get(aclnn, set()))
        ops_all = sorted(set(ops_km) | set(ops_pb) | set(ops_km_gen) | set(ops_pb_gen))

        classes = sorted(km_classes.get(aclnn, set()))
        funcs = sorted(pb_funcs.get(aclnn, set()))
        km_file_list = sorted(km_files.get(aclnn, set()))
        pb_file_list = sorted(pb_files.get(aclnn, set()))

        km_src_hit = bool(ops_km)
        pb_src_hit = bool(ops_pb)
        km_gen_hit = bool(ops_km_gen)
        pb_gen_hit = bool(ops_pb_gen)

        evidence_parts = []
        if km_src_hit:
            evidence_parts.append("kernel_mod_src")
        if pb_src_hit:
            evidence_parts.append("pyboost_src")
        if km_gen_hit:
            evidence_parts.append("kernel_mod_gen")
        if pb_gen_hit:
            evidence_parts.append("pyboost_gen")
        has_custom_dispatch = bool(ops_custom_dispatch)
        if has_custom_dispatch:
            evidence_parts.append("custom_dispatch")
        evidence = "+".join(evidence_parts)

        status = "✅" if evidence_parts else "✖️"
        pyboost_status = "✅" if (pb_src_hit or pb_gen_hit or has_custom_dispatch) else "✖️"
        kbk_status = "✅" if (km_src_hit or km_gen_hit or has_custom_dispatch) else "✖️"

        suffix = aclnn[len("aclnn") :] if aclnn.startswith("aclnn") else aclnn
        suffix_norm = _normalize_name(suffix)
        direct_match_ops = [op for op in ops_all if _normalize_name(op) == suffix_norm]
        suspected_fusion = str(bool((km_src_hit or pb_src_hit or km_gen_hit or pb_gen_hit) and not direct_match_ops))

        remarks: list[str] = []
        if pb_src_hit:
            remarks.append("pyboost_op_inferred_from_filename")
        if len(ops_all) > 1:
            remarks.append(f"shared_by_{len(ops_all)}_ops")
        if ops_custom_dispatch:
            remarks.append("has_custom_dispatch")

        rows.append(
            {
                "aclnn_api": aclnn,
                "mindspore": status,
                "pyboost": pyboost_status,
                "kbk": kbk_status,
                "kernel_mod_src": "✅" if km_src_hit else "✖️",
                "pyboost_src": "✅" if pb_src_hit else "✖️",
                "kernel_mod_gen": "✅" if km_gen_hit else "✖️",
                "pyboost_gen": "✅" if pb_gen_hit else "✖️",
                "evidence": evidence,
                "via_ops": _join_limit(ops_all, limit=12),
                "via_kernel_mod_src": _join_limit(ops_km, limit=12),
                "via_pyboost_src": _join_limit(ops_pb, limit=12),
                "via_kernel_mod_gen": _join_limit(ops_km_gen, limit=12),
                "via_pyboost_gen": _join_limit(ops_pb_gen, limit=12),
                "custom_dispatch_ops": _join_limit(ops_custom_dispatch, limit=12),
                "custom_dispatch_names": _join_limit(dispatch_names, limit=8),
                "direct_match": _join_limit(direct_match_ops, limit=8),
                "suspected_fusion": suspected_fusion,
                "cpp_funcs": _join_limit(classes + funcs, limit=10),
                "kernel_mod_files": _join_limit([_short_path(p) for p in km_file_list], limit=6),
                "pyboost_files": _join_limit([_short_path(p) for p in pb_file_list], limit=6),
                "remarks": ";".join(remarks),
            }
        )

    fieldnames = [
        "aclnn_api",
        "mindspore",
        "pyboost",
        "kbk",
        "kernel_mod_src",
        "pyboost_src",
        "kernel_mod_gen",
        "pyboost_gen",
        "evidence",
        "via_ops",
        "via_kernel_mod_src",
        "via_pyboost_src",
        "via_kernel_mod_gen",
        "via_pyboost_gen",
        "custom_dispatch_ops",
        "custom_dispatch_names",
        "direct_match",
        "suspected_fusion",
        "cpp_funcs",
        "kernel_mod_files",
        "pyboost_files",
        "remarks",
    ]
    write_csv(args.out_csv, rows, fieldnames)
    write_md_table(
        args.out_md,
        rows,
        columns=[
            ("ACLNN API", "aclnn_api"),
            ("mindspore", "mindspore"),
            ("pyboost", "pyboost"),
            ("kbk", "kbk"),
            ("kernel_mod(src)", "kernel_mod_src"),
            ("pyboost(src)", "pyboost_src"),
            ("kernel_mod(gen)", "kernel_mod_gen"),
            ("pyboost(gen)", "pyboost_gen"),
            ("evidence", "evidence"),
            ("via (MS ops)", "via_ops"),
            ("via kernel_mod(src)", "via_kernel_mod_src"),
            ("via pyboost(src)", "via_pyboost_src"),
            ("via kernel_mod(gen)", "via_kernel_mod_gen"),
            ("via pyboost(gen)", "via_pyboost_gen"),
            ("custom dispatch ops", "custom_dispatch_ops"),
            ("direct match", "direct_match"),
            ("suspected fusion", "suspected_fusion"),
            ("C++ funcs", "cpp_funcs"),
            ("kernel_mod files", "kernel_mod_files"),
            ("pyboost files", "pyboost_files"),
            ("remarks", "remarks"),
        ],
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
