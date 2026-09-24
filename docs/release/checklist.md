# 模型发布清单

- [ ] 权重、代码、模型与 processor revision 固定并给出 SHA-256
- [ ] 模型卡说明架构、用途、限制、许可证与未验证能力
- [ ] 数据卡列出来源、配额、许可、过滤、语言、split、防泄漏与已知污染
- [ ] 训练配方含 seed、完整 batch、token/K 分布、硬件、版本、时长和能耗口径
- [ ] 判断、校准、性能、区域 proposal、闭环分别报告并带置信区间
- [ ] 至少一个独立环境复现 run
- [ ] 失败子集、域外风险、policy/confidence 区别写入文档
- [ ] 推理示例不依赖测试答案或隐式外部 agent
- [ ] checkpoint 中移除 optimizer、隐私轨迹、令牌与不必要训练数据
- [ ] 从干净环境完成 data → SFT → eval → release 教程演练
