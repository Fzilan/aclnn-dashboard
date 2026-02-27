#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
RUN_TIMESTAMP="$(TZ=Asia/Shanghai date +"%Y-%m-%dT%H:%M:%S+08:00")"

OP_PLUGIN_ROOT=""
MINDSPORE_ROOT=""
SKIP_SCRAPE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --op-plugin-root)
      OP_PLUGIN_ROOT="$2"
      shift 2
      ;;
    --mindspore-root)
      MINDSPORE_ROOT="$2"
      shift 2
      ;;
    --skip-scrape)
      SKIP_SCRAPE=1
      shift
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${OP_PLUGIN_ROOT}" || -z "${MINDSPORE_ROOT}" ]]; then
  echo "Usage: run_pipeline.sh --op-plugin-root <path> --mindspore-root <path> [--skip-scrape]" >&2
  exit 1
fi

if [[ "${SKIP_SCRAPE}" -eq 0 ]]; then
  python3 scripts/crawl/aclnn_scrape_op_api_list.py --output data/raw/aclnn-aa.generated.md
  python3 scripts/crawl/clean_aclnn_md_table.py data/raw/aclnn-aa.generated.md --inplace
fi

python3 scripts/scan/aclnn_to_torch_npu_report.py \
  --aclnn-md data/raw/aclnn-aa.generated.md \
  --op-plugin-yaml "${OP_PLUGIN_ROOT}/op_plugin/config/op_plugin_functions.yaml" \
  --opapi-src-root "${OP_PLUGIN_ROOT}/op_plugin/ops/opapi" \
  --out-csv data/reports/aclnn_to_torch_npu.csv \
  --out-md data/reports/aclnn_to_torch_npu.md

python3 scripts/scan/aclnn_to_mindspore_report.py \
  --aclnn-md data/raw/aclnn-aa.generated.md \
  --ms-root "${MINDSPORE_ROOT}/mindspore/ops/kernel/ascend/aclnn" \
  --op-def-root "${MINDSPORE_ROOT}/mindspore/ops/op_def/yaml" \
  --aclnn-config "${MINDSPORE_ROOT}/mindspore/python/mindspore/ops_generate/pyboost/aclnn_config.yaml" \
  --out-csv data/reports/aclnn_to_mindspore.csv \
  --out-md data/reports/aclnn_to_mindspore.md

python3 scripts/scan/aclnn_merge_report.py \
  --torch-npu-csv data/reports/aclnn_to_torch_npu.csv \
  --mindspore-csv data/reports/aclnn_to_mindspore.csv \
  --out-csv data/reports/aclnn_to_all.csv \
  --out-md data/reports/aclnn_to_all.md

python3 scripts/build/build_dashboard_data.py --run-timestamp "${RUN_TIMESTAMP}" --output data.json
python3 scripts/build/update_coverage_history.py \
  --data-json data.json \
  --history-file coverage_history.json
python3 scripts/build/build_dashboard_data.py \
  --run-timestamp "${RUN_TIMESTAMP}" \
  --history-file coverage_history.json \
  --output data.json

echo "Pipeline completed."
