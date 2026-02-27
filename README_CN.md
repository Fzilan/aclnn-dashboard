# 🚀 ACLNN Supported Dashboard

Dashboard: https://fzilan.github.io/aclnn-dashboard/

自动生成 ACLNN 在 Torch-NPU / MindSpore 的接入覆盖情况，包含：

- 每日自动扫描与数据更新
- 覆盖率对比与 7 天接入速度趋势

## 仓库结构

```text
aclnn-dashboard/
├─ .github/workflows/
│  └─ refresh-aclnn-dashboard.yml     # 每日自动刷新数据
├─ data/
│  ├─ raw/
│  │  └─ aclnn-aa.generated.md        # ACLNN 全量清单（抓取+清洗后）
│  └─ reports/
│     ├─ aclnn_to_torch_npu.csv/.md   # Torch-NPU 扫描报告
│     ├─ aclnn_to_mindspore.csv/.md   # MindSpore 扫描报告
│     └─ aclnn_to_all.csv/.md         # 合并对比报告
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
├─ index.html                         # Dashboard 页面
├─ data.json                          # 前端消费数据
├─ coverage_history.json              # 每日覆盖率历史
└─ run_pipeline.sh                    # 一键流水线
```

## 功能说明

1. 抓取 ACLNN 全量列表，并清洗为标准表
2. 分别扫描：
   - `op-plugin`（master）得到 Torch-NPU 覆盖数据
   - `mindspore`（master）得到 MindSpore 覆盖数据
3. 合并生成统一对比报告
4. 构建 `data.json`（metrics + operators + history）
5. 维护 `coverage_history.json`，计算 7 天接入速度（pp/day）

## Requirements

- Python 3.9+
- Python 包：`pyyaml`、`playwright`
- Playwright 浏览器：`chromium`

安装示例：

```bash
python3 -m pip install -U pip pyyaml playwright
python3 -m playwright install chromium
```

## 本地使用

### 环境准备

```bash
git clone https://github.com/Fzilan/aclnn-dashboard.git
cd aclnn-dashboard
```

### 3选1 工作流

1. 扫描 + 合并 + 构建（推荐日常使用）

```bash
bash run_pipeline.sh \
  --skip-scrape \
  --op-plugin-root /path/to/op-plugin \
  --mindspore-root /path/to/mindspore
```

2. 含 ACLNN 抓取的全流程（仅当 ACLNN 官网清单变化时）

```bash
bash run_pipeline.sh \
  --op-plugin-root /path/to/op-plugin \
  --mindspore-root /path/to/mindspore
```

3. 仅重建前端数据（不扫描）

```bash
python3 scripts/build/build_dashboard_data.py \
  --history-file coverage_history.json \
  --output data.json
```

### 本地预览页面

```bash
python3 -m http.server 8000
```

打开：`http://localhost:8000`

## 自动化（CI）

工作流：`.github/workflows/refresh-aclnn-dashboard.yml`

- 调度：每天 `UTC 02:00`
- 自动拉取：
  - `https://gitcode.com/Ascend/op-plugin` 的 `master`
  - `https://gitcode.com/mindspore/mindspore` 的 `master`
- 执行 `run_pipeline.sh --skip-scrape`（默认跳过 ACLNN 官网抓取）
- 产物变更时自动 commit & push
- 流水线完成后可直接在 GitHub Pages 查看：`https://fzilan.github.io/aclnn-dashboard/`

## 前端数据契约

- `data.json.metrics`：全局统计 + 7天速度
- `data.json.operators`：算子明细（torch/mindspore 证据）
- `data.json.history.daily_coverage`：历史序列
- `coverage_history.json`：历史数据源（按天去重）
