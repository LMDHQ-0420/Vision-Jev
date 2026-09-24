# 2026-09-25：完成剩余训练数据管线

## 目标

在不新增人工标注的前提下，先用小批量验证 GUI-Odyssey 与同语言 API 改写，再生成正式配额：8,000 条 GUI-Odyssey Choice，以及 2,000 条 API Choice + 1,000 条 API Noul。

## GUI-Odyssey pilot

- 固定 revision：`71e0e7e2d169c642e7a99c264d82ffecadf67889`
- 固定 seed：`vision-jev-sft`
- pilot：128 条 train step，127 个 episode group
- 自动校验：128/128 schema 通过，128/128 图片存在，空目标文本 0，重复候选文本 0
- 候选数：全部为 4；动作标签包括 CLICK 91、TYPE 15、SCROLL 15、COMPLETE 7
- 结论：通过，允许下载 8,500 条候选并从中按正式配置选择 8,000 条

## API 连通性和限额

- Kimi `kimi-k2.6`：实时 smoke 通过；模型只允许 temperature 0.6
- GLM `glm-5.3-flash`：接口返回 429，原因为账号余额不足或无可用资源包，本轮不可用
- Kimi 国内账号：实测组织上限为 3 RPM；国内 Files/Batch 上传端点返回 500，国际端点不接受国内 key
- 实时逐条 pilot：并发 4 时 12/100；单并发时 9/20，均因 RPM 大量失败，不通过
- grouped pilot：一次请求携带 20 个独立 ID，19/20 最终通过；未通过项为否定题，被否定保护规则拒绝
- 第一版正式生成中途抽查发现用词过度正式，370 条全部移入 quarantine，不进入正式候选
- 收紧为自然、简洁措辞后再次 grouped pilot，19/20 通过；逐条抽查保持原语言和原语义，未改变候选、目标或标签；该路径通过，但正式生成增加 10% 候选缓冲

## 正式生成策略

1. 从已验证的 public 90k manifest 按固定 hash 选 2,200 Choice 与 1,100 Noul 父样本。
2. 稳定配置每次请求最多放 60 个独立样本、单并发，调用间隔至少 21 秒，适配 3 RPM 与长请求并发限制。
3. 不使用 GLM 兜底；失败轮次按 2、4、8……秒指数退让，最多 60 秒。
4. 单次运行在 4.75 小时停止发新请求，保留逐批检查点；再次执行从未完成父样本续跑，避免触及 5 小时上限。
5. 每条输出单独执行 schema、语言、数字、否定、长度和资源检查；不同视觉样本可共享通用题干。
6. 每批成功项立即追加到 `processed/api_rewrite/candidates.jsonl`；重启时按 `parent_sample_id` 跳过已完成项。
7. 最终按固定 seed 选择精确的 2,000 Choice 与 1,000 Noul，并与 public 117k 合并为 120k。

## 运行状态

API 生成已结束：57 次 grouped 请求得到 3,251 条合格候选，其中 Choice 2,171、Noul 1,080。全量父子审计的不可变字段违规为 0，语言均继承为英文；题长比为 min 0.552、p50 1.038、p95 1.405、max 2.118。生成阶段拦截记录包括原句照抄 277、否定变化 58、数字变化 6、长度异常 1；被拒绝项未进入候选。候选文件 SHA-256 为 `f503fe2f7d47a67e8e18df7295f6dde263975488d3d26cf3528b67c1fd285286`。

GUI-Odyssey 下载完成：从 89,174 个可用 train step 中固定选择 8,500 条，得到 8,500 条 canonical Choice、4,362 个 episode group，所有图片存在。canonical SHA-256 为 `8707628ec69fb88123a32865b42ac0a75d08057a6ea88491b48e1812eee402f5`。

public manifest 构建完成：117,000 条，Choice 94,000、Noul 17,000、Score 6,000，无来源短缺，SHA-256 为 `5a3275395e9df1c78ec7a43fcaaabf7fbb540a8a80723eb1528916bcaeac851c`。

最终 manifest 构建完成：从 API 合格候选中固定选取 Choice 2,000、Noul 1,000，与 public 117k 合并为 120,000 条、84,428 个 group。最终题型为 Choice 96,000、Noul 18,000、Score 6,000；全部属于 train split，schema、sample ID、group split、target/option 与图片文件检查通过。SHA-256 为 `d0ddde7a861f0e9588e6dff37156c3593c7c390747aca84b2fbe254b931f13c8`。

## 代码变更

- 新增 API key 安全读取后的同语言改写、自动校验、分组请求、节流和断点续跑。
- 评估 API Batch 路径；因当前国内账号文件端点 500，不开放该 CLI，本轮使用已验证的 grouped realtime。
- GUI-Odyssey 下载并发改为命令行可配置，默认 8。
- 新增 public 117k + API 3k 的最终 manifest 精确拼装器。
- 测试、Ruff 和 mypy 必须在提交前全部通过。
