# Data directory

版本库只保存 schema、样例、来源账本和 manifest 元数据。原始/中间/处理数据分别放在被 `.gitignore` 排除的 `data/raw`, `data/interim`, `data/processed`，或通过环境配置指向外部只读存储。

每行 JSONL 表示一道完整问题。不要把 K 个候选展开成 K 条样本。所有路径应相对于数据根或使用不含凭据的内容寻址 URI。
