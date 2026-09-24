# 可复现性标准

每个可复现实验需要四类身份：代码 revision；model/processor/adapter revision；数据 manifest 与 mixture SHA-256；完整配置 SHA-256。再记录 Python、PyTorch、Transformers、PEFT、CUDA、驱动、内核后端、GPU/CPU、主机数量和拓扑。

随机性记录 Python/NumPy/PyTorch/CUDA/环境 seed；说明 deterministic 设置和已知非确定算子。checkpoint 包含 RNG、optimizer、scheduler、sampler 游标和 policy version。恢复训练必须产生新 run 并通过 `parent_run_id` 指向父运行。

性能复现必须保留输入 K、文本长度、真实视觉 token 分布和预热/计时样本。结果复现不是“数字相近”一句话：记录容差、统计方法、置信区间以及无法复现的切片。
