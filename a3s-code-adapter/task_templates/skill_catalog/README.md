# Skill Catalog

`skill_catalog` 是一个小型内部工具仓库，用来维护可复用的 skill 目录并导出机器可读的技能索引。

当前仓库约定：

- 每个 skill 放在 `skills/<slug>/SKILL.md`
- `scripts/build_catalog.py` 会扫描这些 skill 并生成 `catalog/skills_index.json`
- `tests/test_catalog.py` 会校验导出的基础结构

常见改动包括：

- 扩展 catalog 输出字段
- 校验 skill 元数据或脚本路径
- 增加 benchmark / workflow 说明
- 同步 README 示例与导出格式
