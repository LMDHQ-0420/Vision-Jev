# R0 PPO 教程

R0 从 SFT Choice 头复制 actor，新增 value；冻结 VLM、LoRA 与三个判断头。rollout 每个当前观察仍运行 VLM，但 q/h/region/mask 可放 CPU buffer，供 4 个 PPO epoch 重用。

transition 必须保存 policy revision、candidate IDs 与顺序、mask、采样 action、old_logp、冻结判断 reference distribution、value、reward、terminated、truncated 和最终观察引用。候选不得在 update 时重新生成。

初始配置：8 环境 × 64 步，200k transition，minibatch 64，actor/value LR 1e-4，clip 0.2，gamma 0.99，GAE lambda 0.95，value 0.5，entropy 0.01，reference KL 0.01。smoke 先跑 20k。

奖励：成功 +1.0 一次，真实失败 -0.2 一次，每非终局步 -0.002，明确无效动作额外 -0.05。先 MiniGrid，再本地 MiniWoB++/BrowserGym。最终比较使用未见的配对 seed，报告成功率、成功步数、非法动作、超时、恢复率和 p95 端到端延迟。

`terminated` 与 `truncated` 的 bootstrap 规则必须按任务定义测试；Gym 字段不能机械替代有限时域语义。若 R0 受表示限制，另开 ADR 和 R1 配置，不能悄悄让主干参与更新。
