# Notebook Ops Lab

`notebook_ops_lab` 是一个周报分析小仓库，维护数据分析 notebook 和导出脚本。

约定：

- `analysis/weekly_ops_report.ipynb` 保存分析思路与可视化草稿
- `scripts/export_report.py` 负责把 CSV 汇总成 `outputs/weekly_report.json`
- `tests/test_report.py` 会检查导出结果的关键字段

常见改动包括：

- 增加 anomaly / risk summary
- 让 notebook 与导出脚本保持一致
- 生成 markdown brief
- 增加过滤参数但保持默认报表稳定
