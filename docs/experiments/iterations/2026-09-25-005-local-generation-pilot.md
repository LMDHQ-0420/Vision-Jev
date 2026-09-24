# 2026-09-25-005：27k 本地生成数据 pilot

- 状态：measured
- 配方：`sft-120k-v2.2`
- 生成器：`local-27k-v2.2-r2`

## 目标

在不调用生成 API、不新增人工标注的前提下，先用小批量覆盖 27k 本地数据的全部规则族，检查图片、问题、候选、标签与派生关系，再决定是否运行全量。

## 生成设计

全量目标由 24k 程序/环境题和 3k 本地可验证增强组成：2k 全可见网格动作、6k 本地 GUI、6k 候选集合比较、6k 视觉等级、4k 信息充分性、2k 程序困难候选、1k 程序反事实。困难候选与反事实共享父图、继承 `root_id/group_id`，并由确定性求解器重算标签。

pilot 每族取 8 条基础规模，候选集合额外取 16 个父场景以覆盖两个派生族，共 72 题、56 张独立 PNG。任务分布为 Choice 48、Noul 16、Score 8；候选规模覆盖 2–4、5–8、9–16、17–32 四档。全部为英文原生场景，未执行翻译。

## 质量检查与修正

首轮目视发现信息不足图片直接写出 `OCCLUDED` 和解释性文字，可能形成标签捷径。生成器升级到 r2 后改为无答案文字的纯视觉遮挡纹理，并重新生成 pilot。候选集合同时限制只有受控卡片满足筛选条件，避免“第二便宜”因价格并列或额外合格项产生歧义。

r2 自动检查通过：72/72 schema 合法、56/56 图片存在、全部候选框在图像范围内、Choice/Score target 均存在于候选、父子组一致、API 调用 0、项目人工标注 0。manifest SHA-256 为 `cafdca1827c63c306f3d05016b528cb892007f7c605855a7c9457068ef2fb78f`。

## MiniWoB++ 下载与真实环境 pilot

官方仓库已下载到 `/mnt/sda1/sol_data/vision-jev/environments/miniwob-plusplus`，固定提交 `33c3b4ddef8c6eb67c57a29663d844b1eda7e614`。`vision-jev` 环境安装 MiniWoB++ 1.1.0、Selenium 4.49.0，并由 Selenium Manager 下载匹配的 Chrome for Testing/ChromeDriver 154.0.8037.57。

真实环境 pilot 覆盖 `click-test-2`、`click-button`、`click-link`、`click-tab`，每个环境 2 个 seed，共 8 题。8/8 专家点击均由环境正奖励与终止状态验证，最低奖励 0.9903；schema、截图和候选框检查通过。manifest SHA-256 为 `fb2063dde7e2abf52de1318e9f0dff7d27113e53a15d52e62e2c299efd13f05d`。

目视显示 MiniWoB++ 原生截图固定为 160×210，动作标签可靠，但文字和控件较小、界面风格单一。因此它适合作为 6k GUI 中的真实环境组成，不宜独占全部 GUI 配额；全量前应把 MiniWoB++ 与 1024×768 自建沙箱 GUI 混合，并单独报告原生低分辨率轨道。

## 产物

- pilot：`/mnt/sda1/sol_data/vision-jev/pilots/local-27k-v2.2-pilot.jsonl`
- 报告：`/mnt/sda1/sol_data/vision-jev/pilots/local-27k-v2.2-pilot.generation-report.json`
- 图片：`/mnt/sda1/sol_data/vision-jev/generated/local-27k-v2.2-r2-pilot/images/`
- MiniWoB++ pilot：`/mnt/sda1/sol_data/vision-jev/pilots/miniwob-v1-pilot.jsonl`
- MiniWoB++ 报告：`/mnt/sda1/sol_data/vision-jev/pilots/miniwob-v1-pilot.generation-report.json`
- MiniWoB++ 图片：`/mnt/sda1/sol_data/vision-jev/generated/miniwob-pilot-33c3b4d/images/`

本轮仅完成 pilot，不把 27k 标记为已完成；全量生成需在 pilot 质量确认后执行。
