> This plan is written for the Codex agent to execute implementation tasks step by step.
> We appreciate Codex's technical support and disciplined execution throughout this implementation.

**【Role & Objective】**
基于我本地已经跑通的扫描数据，帮我构建一个“ACLNN 接入数字化看板 (Dashboard)”。计划部署到GitHub Pages。


**【Current State (已有资产)】**
我本地已经完成了核心的扫描与清洗，拥有以下产物：

1. `aclnn-aa.generated.md`：从 CANN 官网解析出的全量 aclnn 算子清单。
2. `tools/aclnn_to_torch_npu.csv / .md`：全量 aclnn 算子清单出发，Python 扫描 op_plugin 得到的 Torch 侧算子支持度、证据及详情。
3. `tools/aclnn_to_mindspore.csv / .md`：全量 aclnn 算子清单出发，Python 扫描 MindSpore master 得到的 MS 侧算子支持度、证据及详情。
4. `tools/aclnn_to_all.csv / .md`：以 `aclnn` 为主键合并了上述torch_npu和mindspore 对aclnn 的支持数据，表示aclnn在两个框架的覆盖率对比，markdown 表头出有输出基础统计。


怎么得到这些产物，涉及到的关键 python 脚本：

- `tools/aclnn_scrape_op_api_list.py`：抓取 hiascend 的 文档 `op_api_list` 页面，提取全量 `aclnn*` （主要是从这个页面的目录爬到aclnn的字符串），生成标准 Markdown 表，得到全量的aclnn算子列表。
- `tools/clean_aclnn_md_table.py`：清洗 ACLNN Markdown 表，去重并删除无效 API 行（非 `aclnn` + 大写开头）。  
- `tools/aclnn_to_torch_npu_report.py`：以 ACLNN 为主键反查 torch-npu 接入情况，输出证据来源、ATen 接入入口等等。
- `tools/aclnn_to_mindspore_report.py`：以 ACLNN 为主键反查 MindSpore 接入情况，综合 `kernel_mod_impl` 与 `pyboost_impl` 证据，输出对称字段表。  
- `tools/aclnn_merge_report.py`：合并 torch-npu 与 MindSpore 两张 ACLNN 主键表，生成 7 列对比表（md/csv）并给出覆盖统计。



**【Architecture Philosophy (架构指导原则)】**
* **重 Backend 计算 (Python)**：所有的关联查询、重度指标计算（如完成率、高优缺失算子统计）、格式转换，必须在 Python 脚本中完成，最终吐出一个结构极度友好的 `data.json`。
* **轻 Frontend 渲染 (SPA)**：前端只做纯静态页面。使用单文件 `index.html`，引入 Vue 3 (CDN) 和 Tailwind CSS (CDN)。页面加载时只负责 `fetch('data.json')`，并利用 Vue 的 `computed` 进行轻量级的条件过滤和联动展现。


**【Execution Plan: Phase 1 (当前目标)】**
请按以下步骤引导我完成基础 Dashboard 的搭建，每完成一步请等待我的确认：
**Step 1: 改造数据组装脚本 (build_dashboard_data.py)**
* 基于我现有的合并逻辑，编写一个 Python 脚本。
* 读取本地的 MD/CSV 源文件进行合并。注意解读我已有的 4 个 markdown 表格产物。
* **核心要求**：不仅要输出算子明细列表，还要在 Python 层算出核心全局指标（如 `total_ops`, `torch_supported`, `ms_supported`, `both_supported`, `last_update_time`）。
* 最终导出一个标准的 JSON 结构，范例：`{"metrics": {...}, "operators": [...]}`。


**Step 2: 构建轻量前端 (index.html)**
* 编写单文件 HTML。
* 顶部：展示 Python 传过来的 `metrics` 全局指标（进度条、统计卡片）。
* 中部控制台：利用 Vue 提供按算子名/API名的搜索框，以及分类 Tag 和状态的下拉筛选。
* 底部矩阵表：渲染 `operators` 列表。


**Step 3: 交互增强 (Details Drawer)**
* 在前端实现点击表格行时，弹出一个模态框或侧滑抽屉。
* 抽屉内详细展示该算子对应的 `torch_evidence` 和 `ms_evidence`（这些数据已在 JSON 中就绪）。

**【Future Roadmap (保持代码的扩展性)】**
* Phase 2 我们会加入更多维度的代码扫描指标或者框架算子指标，比如追踪 海思 ACLNN cann 版本支持变化、 aclnn 算子在不同框架的dtype支持/device支持/算子性能数据等），并可能通过定时任务自动触发 Python 脚本更新 JSON。请确保 Python 的指标生成模块和前端的展现模块高度解耦。

* 自动化工作流打通，让数据保鲜：
后续希望自动触发部署 (GitHub Actions)，数据发生变更被推送到仓库，GitHub Actions 就会自动触发 Pages 的重新构建。这样，每天看到的就是最新的算子作战地图，而且整个过程是完全自动化的。
