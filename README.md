# 🚀 ACLNN Supported Dashboard

https://fzilan.github.io/aclnn-dashboard/

Automatically generates and visualizes ACLNN support coverage across Torch-NPU and MindSpore, including:

- Daily automated scanning and data refresh
- Coverage comparison and 7-day onboarding speed trend

## Repository Structure

```text
aclnn-dashboard/
├─ .github/workflows/
│  └─ refresh-aclnn-dashboard.yml     # Daily data refresh workflow
├─ data/
│  ├─ raw/
│  │  └─ aclnn-aa.generated.md        # Full ACLNN list (after crawl + cleanup)
│  └─ reports/
│     ├─ aclnn_to_torch_npu.csv/.md   # Torch-NPU scan report
│     ├─ aclnn_to_mindspore.csv/.md   # MindSpore scan report
│     └─ aclnn_to_all.csv/.md         # Merged comparison report
├─ scripts/
│  ├─ crawl/
│  │  ├─ aclnn_scrape_op_api_list.py
│  │  └─ clean_aclnn_md_table.py
│  ├─ scan/
│  │  ├─ aclnn_to_torch_npu_report.py
│  │  ├─ aclnn_to_mindspore_report.py
│  │  └─ aclnn_merge_report.py
│  └─ build/
│     ├─ build_dashboard_data.py
│     └─ update_coverage_history.py
├─ index.html                         # Dashboard page
├─ data.json                          # Frontend-consumable data
├─ coverage_history.json              # Daily coverage history
└─ run_pipeline.sh                    # One-command pipeline
```

## What It Does

1. Crawls the full ACLNN API list and normalizes it
2. Scans codebases independently:
   - `op-plugin` (master) for Torch-NPU coverage
   - `mindspore` (master) for MindSpore coverage
3. Merges both sides into a unified comparison report
4. Builds `data.json` (metrics + operators + history)
5. Maintains `coverage_history.json` and computes 7-day speed (pp/day)

## Requirements

- Python 3.9+
- Python packages: `pyyaml`, `playwright`
- Playwright browser: `chromium`

Install example:

```bash
python3 -m pip install -U pip pyyaml playwright
python3 -m playwright install chromium
```

## Usage

### 1) Full local pipeline

Run in `aclnn-dashboard/`:

```bash
bash run_pipeline.sh \
  --op-plugin-root /path/to/op-plugin \
  --mindspore-root /path/to/mindspore
```

Notes:

- `--op-plugin-root`: path to the `op-plugin` repo root (master)
- `--mindspore-root`: path to the `mindspore` repo root (master)

### 2) Skip crawling, rerun scan + build only

```bash
bash run_pipeline.sh \
  --skip-scrape \
  --op-plugin-root /path/to/op-plugin \
  --mindspore-root /path/to/mindspore
```

### 3) Rebuild frontend data only

```bash
python3 scripts/build/build_dashboard_data.py \
  --history-file coverage_history.json \
  --output data.json
```

### 4) Preview locally

```bash
python3 -m http.server 8000
```

Open: `http://localhost:8000`

## Automation (CI)

Workflow: `.github/workflows/refresh-aclnn-dashboard.yml`

- Schedule: daily at `UTC 02:00`
- Pulls automatically:
  - `https://gitcode.com/Ascend/op-plugin` at `master`
  - `https://gitcode.com/machangwei/mindspore` at `master`
- Runs `run_pipeline.sh`
- Auto commits and pushes when artifacts change

## Frontend Data Contract

- `data.json.metrics`: global metrics + 7-day speed
- `data.json.operators`: per-operator details (Torch/MindSpore evidence)
- `data.json.history.daily_coverage`: historical daily series
- `coverage_history.json`: source of daily snapshots (deduplicated by date)
