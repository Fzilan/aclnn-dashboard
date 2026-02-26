#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TableSlice:
    start: int
    end: int


_PIPE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_VALID_ACLNN_API_RE = re.compile(r"^aclnn[A-Z].*")


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


def _find_first_pipe_table(lines: list[str]) -> TableSlice:
    for i, line in enumerate(lines):
        if not _PIPE_ROW_RE.match(line):
            continue
        header = _split_pipe_row(line)
        if len(header) < 2 or i + 1 >= len(lines):
            continue
        sep = _split_pipe_row(lines[i + 1])
        if len(sep) != len(header) or not _looks_like_separator_row(sep):
            continue
        j = i + 2
        while j < len(lines) and _PIPE_ROW_RE.match(lines[j]):
            j += 1
        return TableSlice(start=i, end=j)
    raise ValueError("No pipe-style markdown table found.")


def clean_aclnn_table_lines(table_lines: list[str]) -> tuple[list[str], dict[str, int]]:
    header = table_lines[0]
    sep = table_lines[1]
    kept: list[str] = [header.rstrip(), sep.rstrip()]

    seen: set[tuple[str, str]] = set()
    stats = {"dropped_invalid": 0, "dropped_duplicate": 0, "kept": 0}

    for line in table_lines[2:]:
        cells = _split_pipe_row(line)
        if len(cells) < 2:
            continue
        tag = cells[0].strip()
        api = cells[1].strip()

        if not _VALID_ACLNN_API_RE.match(api):
            stats["dropped_invalid"] += 1
            continue

        key = (tag, api)
        if key in seen:
            stats["dropped_duplicate"] += 1
            continue
        seen.add(key)
        kept.append(f"| {tag} | {api} |")
        stats["kept"] += 1

    return kept, stats


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean ACLNN markdown table: drop duplicates and invalid aclnn* rows.")
    p.add_argument("input_md", type=Path)
    p.add_argument("-o", "--output", type=Path, default=None, help="Output path (default: in-place).")
    p.add_argument(
        "--inplace",
        action="store_true",
        help="Rewrite the input file in-place (ignored if --output is set).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    src_path: Path = args.input_md
    out_path: Path = args.output or (src_path if args.inplace else src_path.with_suffix(".cleaned.md"))

    text = src_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    table = _find_first_pipe_table(lines)

    cleaned_table_lines, stats = clean_aclnn_table_lines(lines[table.start : table.end])

    new_lines = lines[: table.start] + cleaned_table_lines + lines[table.end :]
    out_path.write_text("\n".join(new_lines) + ("\n" if text.endswith("\n") else ""), encoding="utf-8")

    print(
        f"kept={stats['kept']} dropped_invalid={stats['dropped_invalid']} dropped_duplicate={stats['dropped_duplicate']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

