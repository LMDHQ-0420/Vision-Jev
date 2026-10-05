# RLCD-inspired 交互场景计划

更新日期：2026-10-03。

## 定位

迷宫、钥匙门、动态避障和推箱子比静态问答更接近 Vision-JEV 的 policy/value 目标。交互轨单独计数，不把环境生成状态冒充 `rlcd_72k` 的公开静态根问题，也不改变 SFT 数据政策。机器可读配额见 `configs/data/rlcd_interactive_16k.json`。

需要区分三种量：

- `decision_probs`：一个动作是否属于 oracle 最优动作集合的校准概率，可用于 RLCD-inspired 离线训练。
- `policy_probs`：为最大化累计回报而选择动作的策略分布，不能命名为 confidence。
- `value`：从当前状态出发的期望归一化回报，由独立 value head 学习，不强行编码成五级图像质量 Score。

## 场景优先级

| 阶段 | 环境 | 交互意义 | 标签方式 |
|---|---|---|---|
| I0 | MiniGrid Empty / FourRooms / MultiRoom | 视觉导航、方向控制、路径长度 | BFS/A* 给出最短路和全部并列最优首动作 |
| I1 | DoorKey / Unlock / Pickup | 工具获取、动作前置条件、访问控制 | 状态搜索验证拿钥匙、开门、到达目标的动作序列 |
| I2 | LavaCrossing / DynamicObstacles | 风险规避、局部可观测、时序决策 | 2,250 条静态危险用精确规划器；250 条动态危险对每个首动作执行 64 次 rollout |
| I3 | BabyAI GoTo / Pickup / Open | 语言指令落地到视觉动作 | 使用环境 mission 与 bot/oracle 轨迹，按 mission + seed 分组 |
| I4 | Procgen Maze | 像素风格和地图规模迁移 | 固定 0.10.7；训练与未见 seed source-shift 分轨 |
| I5 | Boxoban | 不可逆操作、死锁和长程规划 | 固定 Apache-2.0 关卡；先结构化关卡，再接可靠求解器 |

MiniGrid 提供可调尺寸和复杂度的 Empty、DoorKey、Dynamic Obstacles、MultiRoom、Obstructed Maze 以及 BabyAI 等环境，适合课程学习。Procgen Maze 使用程序生成迷宫，地图从 3x3 到 25x25，可测试视觉域变化。Boxoban 有大规模 10x10 关卡和 medium/hard 难度，固定归档内含 Apache-2.0 许可证。

## 16k 离线决策集

MiniGrid/BabyAI 作为基础 12k，Procgen Maze 和 Boxoban 各增加 2k：

| 场景族 | Choice | Noul | 合计 |
|---|---:|---:|---:|
| 基础导航 | 3,200 | 800 | 4,000 |
| 钥匙、门与拾取 | 2,400 | 600 | 3,000 |
| 熔岩与动态障碍 | 1,600 | 900 | 2,500 |
| BabyAI 指令执行 | 2,000 | 500 | 2,500 |
| Procgen Maze | 1,600 | 400 | 2,000 |
| Boxoban | 1,600 | 400 | 2,000 |
| **合计** | **12,400** | **3,600** | **16,000** |

角色分配为 train 10,000、dev 1,500、calibration 1,500、threshold 750、audit 750、test 1,500。按 `environment_id + level_seed + mission` 分组；任意 seed 或 Boxoban level 只能属于一个角色。所有非 train 角色使用未见 seed/level。

## 样本定义

每个根决策至少保存：环境版本、环境 ID、level seed、episode ID、step、mission、RGB observation、有限历史、候选动作、可执行性 mask、oracle Q、全部最优 target action、剩余最短距离、horizon、终止原因和生成代码 revision。

为避免用启发值伪装完整 Q，状态搜索只给所有已证明的并列最优动作写精确
`oracle_q=-shortest_distance`，未继续展开到各自最短成功距离的非最优动作写 `null`，并用
`oracle_q_coverage=all_optimal_actions_exact_nonoptimal_actions_not_expanded` 明示覆盖范围。

Choice 不能用固定 tie-break 把多条同长路径伪装成唯一答案。现有 schema 允许 target 为候选 ID 列表，训练使用 `choice_set_loss` 汇总所有有效目标的概率质量。

Noul 使用可验证命题，例如“执行候选动作后，是否仍存在一条在剩余步数内完成任务的路径”。确定性环境用完整搜索得出真假。动态环境只生成 Choice，问题明确限定为固定反应式策略下 64 次独立随机 rollout 的实测平均回报；同时保存 `successes/trials`、碰撞、超时、平均回报和 `policy_probs`，并标记 `oracle_verified=false`，不冒充确定性 oracle。

合法动作 mask 只能依赖当前真实可执行性，不能读取未来奖励、oracle 路径或隐藏答案。oracle 可以用于标签和审计，但不能进入模型输入。

## 观察与难度

1. 先做 fully-observed RGB，验证视觉、候选和 oracle 对齐。
2. 再做 partial RGB + 最近动作历史，让相同局部观察在不同隐藏地图下产生真实不确定性。
3. 逐步提高地图大小、房间数、路径长度、门/钥匙数量和动态障碍速度。
4. 保留无解、超出 horizon 和不可恢复死锁状态，用于 Noul、阈值和人工接管评测。

同一 episode 的不同 step 必须保持同组。fully-observed 与 partial-observed 视图也必须保持同一角色，防止地图布局泄漏。

## 闭环评测

在线 rollout 不计入 16k 根决策。固定未见 seed，成对比较 random、oracle、SFT/RLCD decision-guided policy 与 PPO policy，并报告：

- 成功率、归一化回报、相对 oracle regret、路径冗余和超时率；
- 碰撞、熔岩、死锁、无效动作和重复循环；
- 自动执行阈值下的 coverage-risk 曲线与低置信接管收益；
- fully-observed 到 partial-observed、静态到动态、MiniGrid 到 Procgen Maze 的迁移退化；
- 延迟一帧和动作执行失败时的鲁棒性。

实现顺序是 `Empty/FourRooms -> DoorKey -> LavaCrossing -> BabyAI GoTo -> Procgen Maze -> Boxoban`。Procgen 与 Boxoban 先完成可复现资产索引；只有 oracle/solver 验证通过的状态才能进入 16k 决策 manifest。

全部交互数据可在 CPU 上断点分阶段生成：

```bash
scripts/generate_interactive_data.sh --data-root /data/vision-jev --max-workers 32
```

脚本逐阶段校验行数和图像资产，已完成阶段会跳过。MiniGrid/BabyAI 枚举真实环境转移，
Procgen 使用固定 0.10.7 的隔离 Python 3.10 环境和 state restore，Boxoban 使用带死角剪枝的
精确 A*。最终合并器严格检查 16k 总量、role×task 配额、唯一图像+指令状态和 group 泄漏，
并写出 `base-16k.jsonl`、`oracle-report.json`、`seed-leakage-report.json` 与生成状态。
