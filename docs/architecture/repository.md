# 仓库文件架构

```text
configs/                 可审查、可冻结的模型/数据/训练/评测配置
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
src/vision_jev/
  data/                   适配、验证、整题 collator
  model/                  VLM 适配、决策头、loss、区域映射
  runtime/                参考/共享执行、缓存身份、API
  train/                  SFT、rollout、PPO update
  eval/                   指标、校准、profile
tests/                    unit/integration/gpu/model 分层测试
runs/                     每次运行的不可变证据（大文件外置）
artifacts/                本地生成报告；不作为唯一事实来源
```

依赖方向保持 `data/model → runtime/train/eval`，训练代码不能反向改变数据真值或评测 split。`docs/experiments` 解释为什么变化，`runs` 证明实际发生了什么，二者不可互相替代。
