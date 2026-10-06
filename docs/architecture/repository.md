# 仓库文件架构

根目录只保留一个项目入口文档 `README.md`。所有设计、开发、训练、评测、实验
和发布说明统一位于 `docs/`；代码、配置、数据契约和生成资产目录不携带局部
README，避免同一规则在多个入口漂移。

```text
configs/                 可审查、可冻结的模型/数据/训练/评测配置
asset/                   品牌资源和经过审查的最终演示资产
data/
  schemas/               JSON 数据、run、PPO transition 契约
  samples/               可公开的最小合成样例
  manifests/             来源、许可与数据 snapshot 元数据
docs/
  architecture/          系统、接口、代码边界
  data/                  来源、转换、拆分和每次数据搭配
  training/              SFT/PPO 操作手册
  evaluation/            判断、概率、性能和闭环协议
  experiments/           每次代码迭代的人类可读账本
  decisions/             不可轻易逆转的 ADR
  release/               模型卡模板与发布门禁
  development/           环境、工程规范和 showcase 操作说明
scripts/                 下载、生成、训练和顺序流水线入口
showcase/                模型对比、闭环轨迹记录、GIF 渲染和主页发布代码
vision_jev/
  data/                   适配、验证、整题 collator
  model/                  VLM 适配、决策头、loss、区域映射
  runtime/                参考/共享执行、缓存身份、API
  train/                  SFT、rollout、PPO update
  eval/                   指标、校准、profile
tests/                    unit/integration/gpu/model 分层测试
runs/                     每次运行的不可变证据（大文件外置）
```

依赖方向保持 `data/model → runtime/train/eval`，训练代码不能反向改变数据真值或评测 split。`docs/experiments` 解释为什么变化，`runs` 证明实际发生了什么，二者不可互相替代。

## 执行边界

- `configs/` 描述实验，不包含业务逻辑或本机凭据。
- `vision_jev/data` 负责规范化、拆分和 oracle；`vision_jev/model` 负责主干与决策头。
- `vision_jev/train` 只消费冻结 manifest；`vision_jev/eval` 只消费冻结角色和输出。
- `scripts/` 编排阶段与恢复点，不重复实现 Python 领域逻辑。
- `showcase/` 是只读消费者：加载已训练 checkpoint，在固定环境中记录轨迹并渲染，不能修改训练数据、模型或正式测试结果。
- `runs/` 只保存小型不可变证据与外部产物引用；原始数据和 checkpoint 位于外部数据根。

## Showcase 与正式评测

每个可视化场景只冻结一个未见 seed，原始 Qwen 与相同参数规模的 Vision-Jev
共享该 seed、动作空间和步数上限。动画用于解释行为，不用于估计总体性能。
正式闭环 test 使用冻结 split 中的多个未见 seed，并单独报告聚合指标和置信区间。
