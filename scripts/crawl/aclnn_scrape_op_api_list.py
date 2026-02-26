#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Source:
    tag: str
    url: str


DEFAULT_SOURCES: tuple[Source, ...] = (
    Source(
        tag="all",
        url="https://www.hiascend.com/document/detail/zh/canncommercial/850/API/aolapi/operatorlist_00001.html",
    ),
    # Source(
    #     tag="ops-math",
    #     url="https://www.hiascend.com/document/detail/zh/canncommercial/850/API/aolapi/context/ops-math/op_api_list.md",
    # ),
    # Source(
    #     tag="ops-nn",
    #     url="https://www.hiascend.com/document/detail/zh/canncommercial/850/API/aolapi/context/ops-nn/op_api_list.md",
    # ),
    # Source(
    #     tag="ops-cv",
    #     url="https://www.hiascend.com/document/detail/zh/canncommercial/850/API/aolapi/context/ops-cv/op_api_list.md",
    # ),
    # Source(
    #     tag="ops-transformer",
    #     url="https://www.hiascend.com/document/detail/zh/canncommercial/850/API/aolapi/context/ops-transformer/op_api_list.md",
    # ),
)


ACLNN_API_RE = re.compile(r"\baclnn[A-Za-z0-9_]+\b")


def _render_table(rows: Iterable[tuple[str, str]]) -> str:
    lines: list[str] = []
    lines.append("# ACLNN supported list")
    lines.append("")
    lines.append("| Tag | ACLNN API |")
    lines.append("|---|---|")
    for tag, api in rows:
        lines.append(f"| {tag} | {api} |")
    lines.append("")
    return "\n".join(lines)


def _extract_aclnn_apis_from_text(text: str) -> set[str]:
    return set(ACLNN_API_RE.findall(text))


def _scrape_sources_with_playwright(sources: Iterable[Source], timeout_ms: int) -> list[tuple[str, str]]:
    try:
        from playwright.sync_api import sync_playwright  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: playwright. Install with:\n"
            "  python -m pip install playwright\n"
            "  python -m playwright install chromium\n"
        ) from exc

    rows: set[tuple[str, str]] = set()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            for src in sources:
                page.goto(src.url, wait_until="networkidle", timeout=timeout_ms)
                text = page.evaluate("() => document.body && document.body.innerText ? document.body.innerText : ''")
                apis = _extract_aclnn_apis_from_text(text or "")
                if not apis:
                    raise RuntimeError(
                        f"No aclnn APIs found on page: {src.url}\n"
                        "Tip: open the URL in a browser and confirm the list is visible. "
                        "If it is, this usually means the page content is blocked or not fully rendered."
                    )
                for api in apis:
                    rows.add((src.tag, api))
        finally:
            browser.close()

    return sorted(rows, key=lambda x: (x[0], x[1]))


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scrape ACLNN operator API lists from hiascend op_api_list pages and output a markdown table.\n\n"
            "Default sources are the 4 categories:\n"
            "  ops-math, ops-nn, ops-cv, ops-transformer"
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write markdown to this path (default: stdout).",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=60_000,
        help="Per-page navigation timeout for Playwright (default: 60000).",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help=(
            "Extra source in form TAG=URL. Can be repeated. "
            "Example: --source ops-math=https://.../op_api_list.md"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)

    sources: list[Source] = list(DEFAULT_SOURCES)
    for item in args.source:
        if "=" not in item:
            raise SystemExit(f"Invalid --source {item!r}, expected TAG=URL")
        tag, url = item.split("=", 1)
        sources.append(Source(tag=tag.strip(), url=url.strip()))

    rows = _scrape_sources_with_playwright(sources, timeout_ms=int(args.timeout_ms))
    md = _render_table(rows)

    if args.output is None:
        sys.stdout.write(md)
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(md, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

