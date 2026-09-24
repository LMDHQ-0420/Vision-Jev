# 接口与概率语义

## 判断模式

- `choice_probs`：在当前有效候选集合内归一化。拒答与信息不足必须是显式、不同的候选。
- `noul_probability`：命题为真的判断概率；0.5 不自动表示“不知道”。
- `score_probs`：每个完整等级描述的概率；加权均值不是成功概率。

## 策略模式

`policy_probs` 表示为了累计回报而选择动作的分布，禁止命名为 `confidence`。合法性 mask 只能基于真实可执行性，不能读取答案、未来奖励或隐藏环境状态。全部 mask 时返回结构化错误。

公共响应至少带 `mode`, `model_revision`, `head_revision`, `processor_revision`, `visual_budget`, `calibration_id`。策略响应另带 `policy_revision`；判断响应不能隐式继承策略校准。

## 形状

问题表示 `[B,1024]`，候选 `[B,K,1024]`，mask `[B,K]`，可选区域 `[B,K,1024]`，geometry `[B,K,6]`。padding 不参加 attention、softmax 或 loss。候选 ID 仅做输出映射，不作为顺序特征。
