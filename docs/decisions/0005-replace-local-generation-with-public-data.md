# ADR-0005：用公开数据替代全部本地生成数据

- 状态：Accepted
- 日期：2026-09-25

## 决策

从 `sft-120k-v2.3` 起，正式训练中本地生成数据为 0。原 27k 配额替换为 GUI-Odyssey 8k、Visual7W 6k、ScienceQA 4k、NLVR 3k、KonIQ-10k 6k。项目只对上游数据执行下载、转换、过滤、去重、分组和抽样，不创建新图片、页面、题目或人工标签。

90k 公开核心保持不变；新增公开数据使公开块达到 117k。其余 3k 使用 API 对已有 Choice/Noul 样本做同语言语义改写，并继承或自动验证标签。完整 120k 仍为 96k Choice、18k Noul、6k Score，项目新增人工标注为 0。

## 后果

活动 CLI 移除 `data-generate-local`。已完成的 local-27k 与 MiniWoB 小批 pilot 不删除，以保证实验历史可追溯，但不进入 canonical pool、manifest 或训练。发布时按来源分别声明许可；包含 ScienceQA 的构建不得描述为无限制商用，KonIQ 原始图片的再发布在条款复核前保持阻断。
