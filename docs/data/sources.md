# 数据来源账本（方案 v2.0）

本表记录清洗后训练目标，不把计划数写成已构建统计。机器可读下载入口、固定 revision 与许可状态见 `configs/data/sources.json`；实际完成状态只认数据盘中的 `_state/downloads/*.json`。

| 来源 | 训练目标 | 首版转换重点 | 许可/发布门禁 |
|---|---:|---|---|
| Multimodal-Mind2Web | 6,000 | 不重复 `action_uid`；仅动作前截图、任务、已发生历史；DOM 候选 | OpenRAIL/研究声明，复核 |
| AndroidControl | 16,000 | 高层目标 + 当前截图；低层当前指令不进主轨；accessibility tree 仅产候选/验证 | 官方研究数据条款复核 |
| WebLINX | 6,000 | 原始截图/页面/步骤对齐；只保留当前点之前历史；接受上游开源发布内容 | CC-BY-NC-SA-4.0，研究版本 |
| GQA | 18,000 | 10k Choice + 8k Noul；balanced 起点；否定题额外核验 | 来源与底图条款复核 |
| RefCOCO/+/g | 18,000 | 初始各 6k；oracle 与 detector 分轨；RefCOCOg 用 UMD split | 标注与 COCO 图片分别复核 |
| VQAv2 | 8,000 | 3.2k Choice + 4.8k Noul；唯一答案要求规范化后 ≥8/10 | 标注与 COCO 图片分别复核 |
| TextVQA | 6,000 | 4.8k Choice + 1.2k Noul；原图为输入，OCR 仅验证/分析 | CC-BY-4.0 |
| ChartQA | 6,000 | 人工问题优先；表格只作 verifier；机器题保留生成方式 | 数据仓库 GPL-3.0，发布复核 |
| CLEVR | 4,000 | 计数、比较、程序诊断 | CC-BY-4.0 |
| MNLI | 2,000 | 纯文本三态；`image=null` | 混合上游条款复核 |

公开来源合计 90,000。SST 已从主训练移除；已下载快照仅为来源追溯，不进入 v2 manifest。程序/环境 24,000 与可核验强模型增强 6,000 的细分见 `docs/data/mixtures.md`。

## 下载状态口径

- 数据根目录固定为 `/mnt/sda1/sol_data/vision-jev`，原始数据永不提交 Git。
- HTTP 文件先写 `.part`，同时验证响应长度、目录中固定的精确字节数和 ZIP 结构；坏包移到带时间戳的 `.invalid-*`，不直接删除。
- Hugging Face 来源固定 commit revision；AndroidControl 当前使用固定 revision 的原始 TFRecord 镜像，并要求发布前与官方 GCS 清单/哈希交叉核验。
- WebLINX 不下载全量 659,934 文件：固定 train CSV 后确定性抽样，只物化入选 turn 的 replay/截图。按项目所有者决定信任上游开源发布的完整内容，不增加 OCR、文本 PII 过滤或发布阻断。
- “已下载”只表示字节和结构完成，不表示许可、隐私、清洗或可训练性已验收。

训练外另建 dev 6,000、calibration/threshold/audit 6,000（3k/2k/1k）和内部 test 12,000，整组隔离且不挪用官方测试样本补训练缺口。
