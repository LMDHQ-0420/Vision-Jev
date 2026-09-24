# 2026-09-24-003：移除 GitHub Actions CI

- 状态：measured
- 变更类型：仓库自动化

## 原因

按项目所有者要求删除 `.github/workflows/ci.yml`。仓库不再在 push 或 pull request 时自动运行 GitHub Actions，也不会再由该工作流触发成功或失败通知邮件。

## 影响

Ruff、格式检查、mypy、测试和仓库契约检查仍可在本地 `vision-jev` 环境运行，但不再由 GitHub 自动执行。删除后检查 `.github/workflows/`，确认不存在其他 workflow 文件。
