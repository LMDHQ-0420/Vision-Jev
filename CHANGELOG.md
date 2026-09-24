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
- 数据策略 v2.3：停止本地生成，原 27k 配额由 GUI-Odyssey、Visual7W、ScienceQA、NLVR、KonIQ-10k 五个公开数据集替代；117k 公开数据自动处理、3k API 同语言改写、0 项目人工标注。
- 从活动 CLI 移除 `data-generate-local`，历史 pilot 仅保留作实验追溯且不进入正式训练清单。
- 新增原始 NLVR canonical 适配器；保留六个官方图像排列并共享 statement group，实测转换 80,394 条且图片/schema 校验通过。
- 新增 Visual7W pointing canonical 适配器，使用上游四候选框、确定性打乱答案位置并按原图分组；实测转换 188,068 条且答案位置近似均匀。
- 新增 ScienceQA 原生多模态题 canonical 适配器；排除 lecture/solution 防止答案泄漏，实测转换 10,332 条且图片/schema 校验通过。
- 新增 GUI-Odyssey 聚合标注、官方路径索引和选择性截图物化管线；64 条公开数据 pilot 转换通过，并修复候选大小写捷径。
- 新增 KonIQ-10k 五级相对质量 Score 适配器；由 train MOS 五等分位产生均衡序数目标，保留全部上游评分统计。
- 新增五来源公开扩展组合 pilot 配置；实测构建 1,964 条且来源、题型、图片和 schema 配额全部通过。

### Changed

- 重写中英文项目首页，增加开放训练链路说明、快速开始和证据驱动 TODO。
- 活动配置与 CLI 使用稳定名称，开发版包标识改为正式包标识。

### Removed

- 删除停用的本地数据生成实现、MiniWoB++ 依赖和对应测试。
- 删除 `CONTRIBUTING.md` 与 `SECURITY.md`。
