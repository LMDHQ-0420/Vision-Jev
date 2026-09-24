# 数据管线

```text
source registration → immutable raw hash → adapter → normalized JSONL
→ quality quarantine → group/deduplicate → split freeze → candidate construction → validators
→ mixture manifest → run snapshot
```

1. `register`：记录来源 URL、版本、许可、责任人和原始 hash；不明许可状态为 `blocked`。
2. `adapt`：每个来源用独立转换器，输出 v2 整题 schema，保留 upstream ID、`root_id`、原始字段映射、标签生成方式和转换器 revision。
3. `group`：先跨来源按图像 hash/original ID 合并，再按网站、模板、seed、trace、翻译/反事实家族建组。
4. `split`：整组分配；最终测试冻结后只运行一次。teacher-only 不得进入 calibration/test。
5. `construct`：同类型难负例、候选顺序随机化；答案位置、K、source 不得形成捷径。
6. `validate`：schema、文件存在、box、ID 唯一、target 映射、group 泄漏、配额、许可、PII 与图像变换。
7. `snapshot`：输出 manifest SHA-256、真实 K/token/language/source 分布，并复制进 run。

答案、`evidence_reference`、教师推理与未来动作不得进入模型输入。GT box 轨道标记 `proposal_kind=oracle`，不能与 detector/DOM 端到端轨道混报。

## 输入防泄漏边界

训练输入必须调用 `build_model_input()` 的显式白名单，只允许图像及其公开变换、状态、问题、允许历史、输入轨道、题型和候选。禁止把整条 canonical JSON 序列化进 prompt。`target`、目标框、未来观测、verifier 结果、教师轨迹、`is_original_target` 等只能位于监督侧。

## 可执行入口

```bash
vision-jev data-download --source multimodal_mind2web
vision-jev data-download --source weblinx --no-extract
vision-jev data-download-weblinx-subset --target-rows 7000
vision-jev data-inventory
vision-jev data-normalize multimodal_mind2web
vision-jev data-build-public --mixture configs/data/sft_120k.json \
  --output /mnt/sda1/sol_data/vision-jev/manifests/public-90k-v2.2.jsonl
```

`data-build-public` 按固定 seed 和 `sample_id` 哈希排序，从每个 canonical train 池取得精确配额；实现使用按配额有界的 streaming heap，不把百万级来源整体载入内存。任一来源不足即写 `*.build-report.json` 并失败，不产出主 manifest；成功时记录最终 manifest SHA-256、题型和语言实测分布。

## v2 来源门禁

- Mind2Web：只接受 CLICK、可合法确定参数的 TYPE/SELECT；动作前截图与 DOM 坐标一致、目标可见；每个原始动作按唯一 `action_uid` 计数。
- AndroidControl：主轨只用高层目标；低层步骤说明另建辅助轨；accessibility tree 不作为纯视觉输入。
- WebLINX：固定 train 索引，确定性选择并只下载所需 replay/截图；信任上游开源发布内容，不增加 OCR/PII 过滤和发布阻断。
- VQAv2/TextVQA：唯一答案主池需规范化后至少 8/10 一致，争议项写 quarantine manifest。
- RefCOCO：oracle/detector 分 manifest 和指标；真实候选未召回目标时记录系统召回失败。
- ChartQA：human/augmented 分开，表格仅 verifier；GQA 否定题做额外证据审计。

## 七层质量流程

结构与文件 → 视觉/坐标 → 标签语义 → 候选与目标 → 泄漏/PII → 去重与整组 split → 配额/分布。每次构建再抽 2% 人工审计；任何失败进入带原因的 quarantine，不静默丢弃。
