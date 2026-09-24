# 实验与代码迭代账本

仓库的每次有意义代码迭代在 `iterations/YYYY-MM-DD-NNN-slug.md` 留档；每次执行在 `runs/<run_id>/` 留机器可读证据。两者必须互相引用。

一个 run 目录至少包含：`run.json`、冻结配置、`data_mixture.json`、环境信息、命令、metrics、summary、checkpoint/artifact hash。`init-run` 只创建新目录，禁止覆盖。无训练的文档/结构检查也建立迭代条目，但明确 `result_status=structural_check`。

复制 `templates/iteration.md` 和 `templates/run-summary.md` 开始。任何失败、无收益与停止条件都记入结论；只记录成功实验会造成选择偏差。
