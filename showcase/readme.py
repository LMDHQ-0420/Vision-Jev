"""Generate the README static RLCD comparison section."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from showcase.schema import ShowcaseConfig

START_MARKER = "<!-- showcase:start -->"
END_MARKER = "<!-- showcase:end -->"
Language = Literal["en", "zh"]

ZH_CAPABILITIES = {
    "general-vqa": "通用视觉问答",
    "text-reading": "自然图像文本读取",
    "chart-reasoning": "图表推理",
    "compositional-reasoning": "组合视觉推理",
    "science-reasoning": "图示科学推理",
    "evidence-judgment": "证据充分性判断",
    "visual-quality": "有序视觉质量评估",
}


def _percent(value: Any) -> str:
    return f"{100 * float(value):.1f}%"


def _result_rows(
    config: ShowcaseConfig, report: dict[str, Any], language: Language
) -> tuple[list[str], list[str]]:
    groups = sorted({model.parameter_group for model in config.models})
    rows = []
    for group in groups:
        summary = report["models"][group]["summary"]
        tasks = summary["by_task"]
        policy = summary["threshold_policy"]
        choice_policy = (
            f"{_percent(policy['choice']['accuracy'])}, 覆盖率 "
            f"{_percent(policy['choice']['coverage'])}"
            if language == "zh"
            else f"{_percent(policy['choice']['accuracy'])} at "
            f"{_percent(policy['choice']['coverage'])} coverage"
        )
        noul_policy = (
            f"{_percent(policy['noul']['accuracy'])}, 覆盖率 {_percent(policy['noul']['coverage'])}"
            if language == "zh"
            else f"{_percent(policy['noul']['accuracy'])} at "
            f"{_percent(policy['noul']['coverage'])} coverage"
        )
        rows.append(
            f"| Vision-Jev-{group} | {_percent(tasks['choice']['accuracy'])} | "
            f"{_percent(tasks['noul']['accuracy'])} | {_percent(tasks['score']['accuracy'])} | "
            f"{choice_policy} | {noul_policy} |"
        )
    return groups, rows


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
        return _fragment_zh(config, groups, rows, asset_root, href)
    return _fragment_en(config, groups, rows, asset_root, href)


def _fragment_en(
    config: ShowcaseConfig,
    groups: list[str],
    rows: list[str],
    asset_root: Path,
    asset_href: str,
) -> str:
    lines = [
        "## Static RLCD results",
        "",
        "The released checkpoints are evaluated on the complete frozen 12,000-question "
        "static RLCD test split. These aggregate results include every success and failure.",
        "",
        "| Model | Choice accuracy | Noul accuracy | Score accuracy | "
        "Choice accepted accuracy | Noul accepted accuracy |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(rows)
    lines.extend(
        [
            "",
            "## Frozen test showcase",
            "",
            "Each animation uses one identical frozen test sample and candidate set for the "
            "original Qwen3.5 checkpoint and Vision-Jev. Categories were declared first. "
            "Within each category, the sample is the minimum SHA-256 sample ID for which both "
            "Vision-Jev checkpoints are correct and pass their already-frozen confidence "
            "threshold; baseline predictions were not used for selection. Each GIF cycles "
            "through every configured example. The two progress bars advance on the measured "
            "median of three post-warmup inference runs, then reveal each model's answer.",
            "",
        ]
    )
    capabilities = ", ".join(example.title for example in config.examples)
    lines.extend([f"Included capabilities: {capabilities}.", ""])
    lines.extend(_showcase_images(groups, asset_root, asset_href, "en"))
    return "\n".join(lines).rstrip() + "\n"


def _fragment_zh(
    config: ShowcaseConfig,
    groups: list[str],
    rows: list[str],
    asset_root: Path,
    asset_href: str,
) -> str:
    lines = [
        "## 静态 RLCD 结果",
        "",
        (
            "已发布 checkpoint 在完整冻结的 12,000 题静态 RLCD 测试集上评测。"
            "以下汇总包含全部成功与失败样本。"
        ),
        "",
        "| 模型 | Choice 准确率 | Noul 准确率 | Score 准确率 | Choice 阈值结果 | Noul 阈值结果 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        *rows,
        "",
        "## 冻结测试集展示",
        "",
        (
            "每段动画都让原始 Qwen3.5 checkpoint 与 Vision-Jev 使用同一个冻结测试样本和"
            "候选集。类别预先确定; 每个类别选择两套 Vision-Jev checkpoint 都回答正确、"
            "通过既定置信阈值且样本 ID 的 SHA-256 最小的样本, 选择过程不使用基线预测。"
            "每张 GIF 会依次播放全部固定样例; 两条进度条按照预热后 3 次推理耗时的中位数"
            "推进, 并在对应模型完成时显示答案。"
        ),
        "",
    ]
    capabilities = "、".join(ZH_CAPABILITIES.get(item.id, item.title) for item in config.examples)
    lines.extend([f"覆盖能力: {capabilities}。", ""])
    lines.extend(_showcase_images(groups, asset_root, asset_href, "zh"))
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
