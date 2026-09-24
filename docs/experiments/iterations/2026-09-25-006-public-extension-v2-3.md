# 2026-09-25-006：公开数据扩展 v2.3

## 触发原因

项目决定停止本地生成，用现有公开数据集补足原 27k 配额，不新增人工标注。

## 本轮变更

- 配方升级为 `sft-120k-v2.3`：117k 公开数据 + 3k API 同语言改写。
- 新增来源：GUI-Odyssey 8k Choice、Visual7W 6k Choice、ScienceQA 4k Choice、NLVR 3k Noul、KonIQ-10k 6k Score。
- 正式配方中的本地生成数量归零，并从活动 CLI 移除生成命令。
- 历史 pilot 保留但明确不参与正式训练。

## 验收口径

下载完成不等于训练验收。每个新来源还需通过：固定版本与来源记录、原始结构检查、canonical 转换、图片存在性、标签映射、按原始 episode/image/presentation 分组、跨来源去重、配额检查和许可发布门禁。小批转换结果及具体统计在实现适配器后追加到本记录。

## 下载进度与问题

- Visual7W pointing 官方标注和 1,860,148,318 字节图像包已下载、校验并解压。canonical 转换实测 188,068 条（train 93,813、dev 36,990、test 57,265），25,733 个图像 group。train 正确答案在四个候选位置的计数为 23,416/23,459/23,493/23,445，未出现答案位置捷径。
- 原始 NLVR 固定提交归档已下载并解压，包含 train/dev/test 标注及对应合成图像。canonical 转换实测为 80,394 条（train 74,460、dev 5,934），13,399 个原始 statement group；train 中 true 41,976、false 32,484，足够从不同 group 自动抽取目标 3k。
- GUI-Odyssey 最初按 7,736 个小 annotation 文件下载，在约 1,000 个时遇到 Hugging Face 429。目录检查确认官方同一 revision 提供 33MB 聚合 `all_anno.json`，因此下载策略改为单文件固定 revision；已缓存的小文件保留但不作为正式入口。截图仍按入选 train episode 选择性物化。
- ScienceQA 固定提交、官方 S3 train/val/test 图像均已下载、校验并解压。canonical 转换实测 10,332 条原生多模态选择题（train 6,218、dev 2,097、test 2,017），足够抽取目标 4k；只保留题目、hint、原始选项和标签，不把 lecture/solution 放入模型输入。
- GUI-Odyssey 聚合标注和 random split 已下载完成，train 中有 89,174 个可用原生动作步骤。建立官方截图路径索引后选择性物化 64 条 pilot（64 个不同 episode，约 38MB），canonical 图片/schema 校验通过。初版 pilot 曾出现目标动作保留大写、负例被规范化为小写的捷径，已在正式记录前修复为全部候选统一规范化。
- KonIQ-10k 10,073 张官方 1024×768 图片和 1.2M 人类评分统计已下载、校验并转换（train 8,102、dev 966、test 1,005）。直接使用五级众数会导致 train 的第 5 类为 0，因此 Score 目标改为仅由 train MOS 计算的五等分位序数等级；train 五类分别为 1,620/1,620/1,619/1,616/1,627，原始 c1-c5、MOS、SD 全部保留。

## 五来源组合 pilot

构建 `/mnt/sda1/sol_data/vision-jev/pilots/public-extension-v2.3.jsonl`，共 1,964 条：GUI-Odyssey 64、Visual7W 600、ScienceQA 400、NLVR 300、KonIQ-10k 600。结果为 Choice 1,064、Noul 300、Score 600，英文 1,964，无来源/题型缺口；SHA-256 为 `9445a7ee5831b3e9b3ced4c92a86dd72bf8de508de37c3204757046a102fe93c`。Noul true/false 为 161/139，Score 五级为 115/147/111/101/126，pilot 未出现缺类。
