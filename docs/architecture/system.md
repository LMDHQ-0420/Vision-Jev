# 系统架构

```text
image ─ native processor ─ vision encoder ─┐
state + question ─ tokenizer ──────────────┼─ Qwen3.5 backbone ─ q/h/readout spans
dynamic candidates ─ serializer ──────────┘                    │
                                  ┌────────────────────────────┼─────────────┐
                                  │                            │             │
                         Choice/Set head                  Noul head      Score head
                                  │
                      frozen copy → PPO actor + value
```

主干始终使用官方 Qwen3.5 多模态实现和 processor。读出 span 在序列化时显式记录，不通过字符串搜索。视觉编码器输出可供区域池化复用；区域坐标必须经历与 processor 完全相同的变换。

参考后端逐候选运行，是正确性基线。共享后端复用公共前缀，但必须复制/隔离 KV、卷积与递推状态。仅 attention mask 不足以隔离混合线性注意力状态。任何权重、processor、裁剪、图像内容或 mode 变化都会使缓存失效。

代码边界：`data` 只负责规范化与整题 batch；`model` 只接收张量契约；`runtime` 管理序列化、缓存与后端；`train` 管理优化状态；`eval` 不改变模型或校准参数。
