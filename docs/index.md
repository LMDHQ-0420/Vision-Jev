# Vision-JEV 教程与研究手册

这套文档同时服务于新贡献者、训练操作者和模型发布审查者。推荐按以下顺序阅读：

1. [产品与研究范围](project-plan.md)
2. [系统架构](architecture/system.md)、[仓库架构](architecture/repository.md) 与 [接口语义](architecture/interfaces.md)
3. [数据来源](data/sources.md)、[数据管线](data/pipeline.md) 与 [混合配方](data/mixtures.md)
4. [SFT 运行手册](training/sft.md)、[PPO 运行手册](training/ppo.md)、[实验追踪](experiments/README.md)
5. [评测协议](evaluation/protocol.md) 与 [发布清单](release/checklist.md)

本机开发环境的实际版本、双 GPU smoke test 与沙箱边界见 [环境基线](development/environment-baseline.md)。

“计划”只表示将做什么；“结构验证”表示在合成输入上检查接口/数学；“实测”必须由 `runs/<run_id>/` 内不可变证据支持。技术计划书是设计输入，不替代代码、运行记录或最终模型卡。
