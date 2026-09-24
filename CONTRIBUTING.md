# 贡献指南

任何代码、数据配方或实验结论都必须可追溯。提交前请阅读 `docs/development/standards.md`。

基本流程：从 issue/研究假设开始；修改配置和代码；创建 run；记录环境、数据组合、命令、结果及失败；更新 `docs/experiments/iterations/` 和 `CHANGELOG.md`；最后运行 `make validate test docs-check`。

禁止提交原始受限数据、访问令牌、大模型权重和含隐私的轨迹。对性能结论必须给出硬件、版本、输入 shape、预热、样本数以及统计口径。负结果与中止原因同样需要记录。

提交标题采用 Conventional Commits：`feat:`, `fix:`, `data:`, `train:`, `eval:`, `docs:`, `infra:`。一个提交只处理一个可审查主题。
