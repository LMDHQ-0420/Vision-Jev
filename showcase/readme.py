"""Generate the README static RLCD comparison section."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from showcase.schema import ShowcaseConfig

START_MARKER = "<!-- showcase:start -->"
END_MARKER = "<!-- showcase:end -->"
Language = Literal["en", "zh"]
TASKS = ("choice", "noul", "score")

ZH_CAPABILITIES = {
    "general-vqa": "通用视觉问答",
    "text-reading": "自然图像文本读取",
    "chart-reasoning": "图表推理",
    "compositional-reasoning": "组合视觉推理",
    "science-reasoning": "图示科学推理",
    "evidence-judgment": "证据充分性判断",
    "visual-quality": "有序视觉质量评估",
}
SOURCE_LABELS = {
    "android_control": "Android Control",
    "chartqa": "ChartQA",
    "clevr": "CLEVR",
    "gqa": "GQA",
    "koniq10k": "KonIQ-10k",
    "multi_nli": "MultiNLI",
    "nlvr": "NLVR",
    "refcoco": "RefCOCO",
    "refcog": "RefCOG",
    "refcocog": "RefCOCOg",
    "refcocoplus": "RefCOCO+",
    "scienceqa": "ScienceQA",
    "textvqa": "TextVQA",
    "visual7w": "Visual7W",
    "vqav2": "VQAv2",
}


def _percent(value: Any) -> str:
    return f"{100 * float(value):.1f}%"


def _accuracy_latency(metric: dict[str, Any]) -> str:
    latency = metric.get("latency_ms")
    mean = "N/A" if latency is None else f"{float(latency['mean_ms']):.1f} ms"
    return f"{_percent(metric['accuracy'])} / {mean}"


def _result_rows(
    config: ShowcaseConfig, report: dict[str, Any], language: Language
) -> tuple[list[str], list[str]]:
    groups = sorted({model.parameter_group for model in config.models})
    rows = []
    for group in groups:
        baseline_summary = report["baselines"][group]["summary"]
        baseline_tasks = baseline_summary["by_task"]
        baseline_model = next(
            model
            for model in config.models
            if model.parameter_group == group and model.role == "baseline"
        )
        rows.append(
            f"| {baseline_model.label} ({'原始' if language == 'zh' else 'original'}) | "
            f"{_percent(baseline_tasks['choice']['accuracy'])} | "
            f"{_percent(baseline_tasks['noul']['accuracy'])} | "
            f"{_percent(baseline_tasks['score']['accuracy'])} |"
        )
        sft_summary = report["sft_models"][group]["summary"]
        sft_tasks = sft_summary["by_task"]
        sft_model = next(
            model
            for model in config.models
            if model.parameter_group == group and model.role == "sft"
        )
        rows.append(
            f"| {sft_model.label} | {_percent(sft_tasks['choice']['accuracy'])} | "
            f"{_percent(sft_tasks['noul']['accuracy'])} | "
            f"{_percent(sft_tasks['score']['accuracy'])} |"
        )
        summary = report["models"][group]["summary"]
        tasks = summary["by_task"]
        rows.append(
            f"| Vision-Jev-{group} | {_percent(tasks['choice']['accuracy'])} | "
            f"{_percent(tasks['noul']['accuracy'])} | {_percent(tasks['score']['accuracy'])} |"
        )
    return groups, rows


def _model_label(config: ShowcaseConfig, group: str, role: str) -> str:
    return next(
        model.label
        for model in config.models
        if model.parameter_group == group and model.role == role
    )


def _validity_lines(
    config: ShowcaseConfig,
    report: dict[str, Any],
    groups: list[str],
    language: Language,
) -> list[str]:
    title = "### 结构化输出有效率" if language == "zh" else "### Structured-output validity"
    lines = [
        title,
        "",
        "| 模型 | Choice | Noul | Score |"
        if language == "zh"
        else "| Model | Choice | Noul | Score |",
        "| --- | ---: | ---: | ---: |",
    ]
    for group in groups:
        for key, role in (("baselines", "baseline"), ("sft_models", "sft")):
            summary = report[key][group]["summary"]
            values = [_percent(summary["by_task"][task]["valid_rate"]) for task in TASKS]
            lines.append(f"| {_model_label(config, group, role)} | " + " | ".join(values) + " |")
        lines.append(f"| {_model_label(config, group, 'trained')} | 100.0% | 100.0% | 100.0% |")
    explanation = (
        "Vision-Jev 通过固定决策头直接返回候选, 因而结构化输出始终有效。"
        if language == "zh"
        else "Vision-Jev returns candidates through fixed decision heads, so its structured "
        "output is always valid."
    )
    lines.extend(["", explanation, ""])
    return lines


def _calibration_lines(
    config: ShowcaseConfig,
    report: dict[str, Any],
    groups: list[str],
    language: Language,
) -> list[str]:
    title = (
        "### Vision-Jev 概率与校准指标"
        if language == "zh"
        else "### Vision-Jev probability and calibration metrics"
    )
    lines = [
        title,
        "",
        "| 模型 | 任务 | 题数 | 准确率 | NLL | Brier | ECE | RPS | MAE |"
        if language == "zh"
        else "| Model | Task | Questions | Accuracy | NLL | Brier | ECE | RPS | MAE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in groups:
        summary = report["models"][group]["summary"]
        label = _model_label(config, group, "trained")
        for task in TASKS:
            metric = summary["by_task"][task]
            lines.append(
                f"| {label} | {task.title()} | {int(metric['questions']):,} | "
                f"{_percent(metric['accuracy'])} | {float(metric['nll']):.3f} | "
                f"{float(metric['brier']):.3f} | {float(metric['ece_15']):.3f} | "
                f"{float(metric['rps']):.3f} | {float(metric['mean_absolute_error']):.3f} |"
                if task == "score"
                else f"| {label} | {task.title()} | {int(metric['questions']):,} | "
                f"{_percent(metric['accuracy'])} | {float(metric['nll']):.3f} | "
                f"{float(metric['brier']):.3f} | {float(metric['ece_15']):.3f} | N/A | N/A |"
            )
    lines.append("")
    return lines


def _threshold_lines(
    config: ShowcaseConfig,
    report: dict[str, Any],
    groups: list[str],
    language: Language,
) -> list[str]:
    lines = [
        "### 阈值策略" if language == "zh" else "### Accepted-set calibration",
        "",
        "| 模型 | 任务 | 阈值 | 覆盖率 | 接受准确率 |"
        if language == "zh"
        else "| Model | Task | Threshold | Coverage | Accepted accuracy |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for group in groups:
        label = _model_label(config, group, "trained")
        policy = report["models"][group]["summary"]["threshold_policy"]
        for task in TASKS:
            metric = policy[task]
            lines.append(
                f"| {label} | {task.title()} | {float(metric['threshold']):.3f} | "
                f"{_percent(metric['coverage'])} | {_percent(metric['accuracy'])} |"
            )
    lines.append("")
    return lines


def _high_confidence_lines(
    config: ShowcaseConfig,
    report: dict[str, Any],
    groups: list[str],
    language: Language,
) -> list[str]:
    lines = [
        "#### 高置信错误" if language == "zh" else "#### High-confidence errors",
        "",
        (
            "统计置信度不低于 0.9 但预测错误的完整测试样本。"
            if language == "zh"
            else "Counts cover every incorrect test prediction with confidence at least 0.9."
        ),
        "",
        "| 模型 | 错误数 | 测试题数 | 占比 |"
        if language == "zh"
        else "| Model | Errors | Test questions | Rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    for group in groups:
        summary = report["models"][group]["summary"]
        errors = int(summary["high_confidence_errors_0_9"])
        questions = int(summary["questions"])
        lines.append(
            f"| {_model_label(config, group, 'trained')} | {errors:,} | {questions:,} | "
            f"{_percent(errors / questions)} |"
        )
    lines.append("")
    return lines


def _timing_lines(
    config: ShowcaseConfig,
    report: dict[str, Any],
    groups: list[str],
    language: Language,
) -> list[str]:
    title = "### 推理耗时" if language == "zh" else "### Inference timing"
    lines = [
        title,
        "",
        (
            "原始 Qwen 与 SFT 的 JSONL 保留每一道题的生成耗时; Vision-Jev 的独立计时"
            "复测保留同步后的完整决策路径耗时。下表由全部逐题记录汇总, 两种口径分开标注。"
            if language == "zh"
            else "The original-Qwen and SFT JSONL files retain generation latency for every "
            "question. The independent Vision-Jev timing runs retain synchronized full "
            "decision-path latency. The table summarizes every per-question record and labels "
            "the two scopes separately."
        ),
        "",
        "| 模型 | 口径 | 总计 (s) | 均值 (ms) | P50 | P95 | P99 | 最小 | 最大 |"
        if language == "zh"
        else "| Model | Scope | Total (s) | Mean (ms) | P50 | P95 | P99 | Min | Max |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    generation_scope = "逐题生成" if language == "zh" else "Per-question generation"
    decision_scope = "完整决策循环" if language == "zh" else "Full decision loop"
    for group in groups:
        for key, role in (("baselines", "baseline"), ("sft_models", "sft")):
            summary = report[key][group]["summary"]
            latency = summary["latency_ms"]
            lines.append(
                f"| {_model_label(config, group, role)} | {generation_scope} | "
                f"{float(latency['total_ms']) / 1000:.1f} | "
                f"{float(latency['mean_ms']):.1f} | {float(latency['p50_ms']):.1f} | "
                f"{float(latency['p95_ms']):.1f} | {float(latency['p99_ms']):.1f} | "
                f"{float(latency['min_ms']):.1f} | {float(latency['max_ms']):.1f} |"
            )
        trained = report["models"][group]["summary"]
        latency = trained.get("latency_ms")
        if latency is None:
            total_seconds = float(trained["elapsed_seconds"])
            mean_ms = total_seconds * 1000 / int(trained["questions"])
            lines.append(
                f"| {_model_label(config, group, 'trained')} | {decision_scope} | "
                f"{total_seconds:.1f} | {mean_ms:.1f} | N/A | N/A | N/A | N/A | N/A |"
            )
        else:
            lines.append(
                f"| {_model_label(config, group, 'trained')} | {decision_scope} | "
                f"{float(latency['total_ms']) / 1000:.1f} | "
                f"{float(latency['mean_ms']):.1f} | {float(latency['p50_ms']):.1f} | "
                f"{float(latency['p95_ms']):.1f} | {float(latency['p99_ms']):.1f} | "
                f"{float(latency['min_ms']):.1f} | {float(latency['max_ms']):.1f} |"
            )
    lines.extend(
        [
            "",
            "<details>",
            "<summary><strong>分任务耗时</strong></summary>"
            if language == "zh"
            else "<summary><strong>Timing by task</strong></summary>",
            "",
            "| 模型 | 任务 | 样本 | 均值 (ms) | P50 | P95 | P99 | 最小 | 最大 |"
            if language == "zh"
            else "| Model | Task | Samples | Mean (ms) | P50 | P95 | P99 | Min | Max |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for group in groups:
        for key, role in (
            ("baselines", "baseline"),
            ("sft_models", "sft"),
            ("models", "trained"),
        ):
            summary = report[key][group]["summary"]
            for task in TASKS:
                latency = summary["by_task"][task].get("latency_ms")
                if latency is None:
                    values = "N/A | N/A | N/A | N/A | N/A | N/A | N/A"
                else:
                    values = (
                        f"{int(latency['samples']):,} | {float(latency['mean_ms']):.1f} | "
                        f"{float(latency['p50_ms']):.1f} | {float(latency['p95_ms']):.1f} | "
                        f"{float(latency['p99_ms']):.1f} | {float(latency['min_ms']):.1f} | "
                        f"{float(latency['max_ms']):.1f}"
                    )
                lines.append(f"| {_model_label(config, group, role)} | {task.title()} | {values} |")
    lines.extend(["", "</details>", ""])
    return lines


def _source_lines(
    config: ShowcaseConfig,
    report: dict[str, Any],
    group: str,
    language: Language,
) -> list[str]:
    title = (
        f"{group} 分数据集与任务结果"
        if language == "zh"
        else f"{group} results by dataset and task"
    )
    baseline_label = _model_label(config, group, "baseline")
    sft_label = _model_label(config, group, "sft")
    trained_label = _model_label(config, group, "trained")
    lines = [
        "<details>",
        f"<summary><strong>{title}</strong></summary>",
        "",
        (
            "每个结果单元格依次为准确率 / 单样本平均耗时。"
            if language == "zh"
            else "Each result cell reports accuracy / mean per-sample latency."
        ),
        "",
    ]
    trained_sources = report["models"][group]["by_source"]
    baseline_sources = report["baselines"][group]["summary"]["by_source"]
    sft_sources = report["sft_models"][group]["summary"]["by_source"]
    for task in TASKS:
        task_sources = [
            source for source, tasks in sorted(trained_sources.items()) if task in tasks
        ]
        if not task_sources:
            continue
        lines.extend(
            [
                f"#### {task.title()}",
                "",
                f"| {'数据集' if language == 'zh' else 'Dataset'} | "
                f"{'题数' if language == 'zh' else 'Questions'} | {baseline_label} | "
                f"{sft_label} | {trained_label} |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for source in task_sources:
            trained_metric = trained_sources[source][task]
            baseline_metric = baseline_sources[source]["by_task"][task]
            sft_metric = sft_sources[source]["by_task"][task]
            questions = int(trained_metric["questions"])
            if questions != int(baseline_metric["questions"]) or questions != int(
                sft_metric["questions"]
            ):
                raise ValueError(f"source/task question mismatch for {group} {source} {task}")
            lines.append(
                f"| {SOURCE_LABELS.get(source, source)} | {questions:,} | "
                f"{_accuracy_latency(baseline_metric)} | "
                f"{_accuracy_latency(sft_metric)} | "
                f"{_accuracy_latency(trained_metric)} |"
            )
        lines.append("")
    lines.extend(["</details>", ""])
    return lines


def _showcase_images(
    groups: list[str], asset_root: Path, asset_href: str, language: Language
) -> list[str]:
    lines: list[str] = []
    for group in groups:
        filename = f"{group.lower()}.gif"
        asset = asset_root / filename
        if not asset.is_file():
            raise FileNotFoundError(f"missing showcase GIF: {asset}")
        if language == "zh":
            alt = f"{group}: 原始 Qwen3.5 与 Vision-Jev 冻结 RLCD 样例对比"
        else:
            alt = f"{group}: original Qwen3.5 versus Vision-Jev across frozen RLCD examples"
        lines.extend(
            [
                f"### {group}",
                "",
                f'<p align="center"><img src="{asset_href.rstrip("/")}/{filename}" '
                f'alt="{alt}" width="900"></p>',
                "",
            ]
        )
    return lines


def _showcase_section(
    config: ShowcaseConfig,
    groups: list[str],
    asset_root: Path,
    asset_href: str,
    language: Language,
) -> list[str]:
    if language == "zh":
        description = (
            "每段动画都让原始 Qwen3.5 checkpoint 与 Vision-Jev 使用同一个冻结测试样本和"
            "候选集。类别预先确定; 每个类别选择两套 Vision-Jev checkpoint 都回答正确、"
            "通过既定置信阈值且样本 ID 的 SHA-256 最小的样本, 选择过程不使用基线预测。"
            "每张 GIF 会依次播放全部固定样例; 两条进度条按照预热后 3 次推理耗时的中位数"
            "推进, 并在对应模型完成时显示答案。"
        )
        capabilities = "、".join(
            ZH_CAPABILITIES.get(item.id, item.title) for item in config.examples
        )
        lines = ["## 冻结测试集展示", "", description, "", f"覆盖能力: {capabilities}。", ""]
    else:
        description = (
            "Each animation uses one identical frozen test sample and candidate set for the "
            "original Qwen3.5 checkpoint and Vision-Jev. Categories were declared first. "
            "Within each category, the sample is the minimum SHA-256 sample ID for which both "
            "Vision-Jev checkpoints are correct and pass their already-frozen confidence "
            "threshold; baseline predictions were not used for selection. Each GIF cycles "
            "through every configured example. The two progress bars advance on the measured "
            "median of three post-warmup inference runs, then reveal each model's answer."
        )
        capabilities = ", ".join(example.title for example in config.examples)
        lines = [
            "## Frozen test showcase",
            "",
            description,
            "",
            f"Included capabilities: {capabilities}.",
            "",
        ]
    lines.extend(_showcase_images(groups, asset_root, asset_href, language))
    return lines


def _aggregate_lines(groups: list[str], rows: list[str], language: Language) -> list[str]:
    lines: list[str] = []
    for index, group in enumerate(groups):
        lines.extend(
            [
                f"### {group} {'汇总' if language == 'zh' else 'summary'}",
                "",
                "| 模型 | Choice | Noul | Score |"
                if language == "zh"
                else "| Model | Choice | Noul | Score |",
                "| --- | ---: | ---: | ---: |",
                *rows[index * 3 : index * 3 + 3],
                "",
            ]
        )
    return lines


def fragment_text(
    config: ShowcaseConfig,
    asset_root: Path,
    test_report: Path,
    *,
    language: Language = "en",
    asset_href: str | None = None,
) -> str:
    report = json.loads(test_report.read_text(encoding="utf-8"))
    groups, rows = _result_rows(config, report, language)
    href = asset_href or asset_root.as_posix()
    if language == "zh":
        return _fragment_zh(config, report, groups, rows, asset_root, href)
    return _fragment_en(config, report, groups, rows, asset_root, href)


def _fragment_en(
    config: ShowcaseConfig,
    report: dict[str, Any],
    groups: list[str],
    rows: list[str],
    asset_root: Path,
    asset_href: str,
) -> str:
    lines = _showcase_section(config, groups, asset_root, asset_href, "en")
    lines.extend(
        [
        "## Static RLCD results",
        "",
        "Original Qwen, the completed SFT-only checkpoint, and Vision-Jev with its static "
        "RLCD decision heads are evaluated on the same complete frozen 12,000-question test "
        "split. These aggregate results include every success and failure, so raw-accuracy "
        "regressions between stages remain visible. Original Qwen and SFT-only generation do "
        "not provide calibrated decision-head probabilities, so accepted-set calibration is "
        "reported only for Vision-Jev.",
        "",
        ]
    )
    lines.extend(_aggregate_lines(groups, rows, "en"))
    lines.extend(_threshold_lines(config, report, groups, "en"))
    lines.extend(_high_confidence_lines(config, report, groups, "en"))
    lines.extend(_timing_lines(config, report, groups, "en"))
    lines.extend(["<details>", "<summary><strong>Calibration details</strong></summary>", ""])
    lines.extend(_validity_lines(config, report, groups, "en"))
    lines.extend(_calibration_lines(config, report, groups, "en"))
    lines.extend(["</details>", ""])
    for group in groups:
        lines.extend(_source_lines(config, report, group, "en"))
    return "\n".join(lines).rstrip() + "\n"


def _fragment_zh(
    config: ShowcaseConfig,
    report: dict[str, Any],
    groups: list[str],
    rows: list[str],
    asset_root: Path,
    asset_href: str,
) -> str:
    lines = _showcase_section(config, groups, asset_root, asset_href, "zh")
    lines.extend(
        [
        "## 静态 RLCD 结果",
        "",
        (
            "原始 Qwen、完成 SFT 但未接入静态 RLCD 决策头的 checkpoint, 以及完整 "
            "Vision-Jev 均在同一份冻结的 12,000 题测试集上评测。以下汇总保留全部成功与"
            "失败样本, 因此不同阶段的原始准确率回退也会直接展示。原始 Qwen 与 SFT-only "
            "生成不提供校准决策头概率, 因此接受集校准只对 Vision-Jev 报告。"
        ),
        "",
        ]
    )
    lines.extend(_aggregate_lines(groups, rows, "zh"))
    lines.extend(_threshold_lines(config, report, groups, "zh"))
    lines.extend(_high_confidence_lines(config, report, groups, "zh"))
    lines.extend(_timing_lines(config, report, groups, "zh"))
    lines.extend(["<details>", "<summary><strong>校准详细指标</strong></summary>", ""])
    lines.extend(_validity_lines(config, report, groups, "zh"))
    lines.extend(_calibration_lines(config, report, groups, "zh"))
    lines.extend(["</details>", ""])
    for group in groups:
        lines.extend(_source_lines(config, report, group, "zh"))
    return "\n".join(lines).rstrip() + "\n"


def build_fragment(
    config: ShowcaseConfig,
    asset_root: Path,
    test_report: Path,
    destination: Path,
    *,
    language: Language = "en",
    asset_href: str | None = None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        fragment_text(
            config,
            asset_root,
            test_report,
            language=language,
            asset_href=asset_href,
        ),
        encoding="utf-8",
    )


def publish_readme(
    config: ShowcaseConfig,
    asset_root: Path,
    test_report: Path,
    readme_path: Path,
    *,
    language: Language = "en",
    asset_href: str | None = None,
) -> None:
    current = readme_path.read_text(encoding="utf-8")
    if current.count(START_MARKER) != 1 or current.count(END_MARKER) != 1:
        raise ValueError(f"{readme_path} must contain one showcase marker pair")
    before, remainder = current.split(START_MARKER, 1)
    _, after = remainder.split(END_MARKER, 1)
    updated = (
        before.rstrip()
        + "\n\n"
        + START_MARKER
        + "\n"
        + fragment_text(
            config,
            asset_root,
            test_report,
            language=language,
            asset_href=asset_href,
        )
        + END_MARKER
        + after
    )
    readme_path.write_text(updated, encoding="utf-8")
