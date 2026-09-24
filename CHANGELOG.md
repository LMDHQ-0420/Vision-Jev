# Changelog

格式遵循 Keep a Changelog；发布版本遵循语义化版本。研究迭代的详细过程见 `docs/experiments/iterations/`。

## [Unreleased]

### Added

- M0 仓库脚手架、数据与运行 schema、配置、研究账本和端到端教程结构。
- CPU 可执行的数据校验、运行初始化/完成命令和仓库一致性检查。
- PyTorch 决策头与损失的首版实现（需安装训练依赖，尚未连接真实 VLM）。
- 数据方案 v2 的固定版本下载、断点/校验、十来源 canonical 适配、确定性配额构建与 WebLINX 选择性截图管线。
- 数据策略 v2.2：WebLINX 直接使用上游开源内容；保留来源语言；不增加 `task_family`；新增 30k 数据按 27k 本地自动生成、3k API 辅助且程序验证、0 项目人工标注规划。
- CLEVR 标签来源修正为程序生成，公共 90k manifest 默认输出升级为 v2.2。
- 新增确定性 `data-generate-local` 管线：视觉网格/GUI/集合比较/Score/信息充分性及求解器验证的困难候选、反事实；支持小批 pilot 后再运行 27k 全量。
- 固定并验证 MiniWoB++ 官方环境，新增真实截图、DOM 候选和环境奖励共同校验的 MiniWoB++ pilot 入口。
