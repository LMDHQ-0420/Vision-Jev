from __future__ import annotations

from vision_jev.data.pipeline import (
    _answer_kind,
    _most_specific_target,
    compatible_answer_pool,
    compatible_answer_pools,
    open_qa_sample,
)


def test_compatible_pool_keeps_question_family_and_answer_kind() -> None:
    rows = [
        {"split": "train", "question": "What color is it?", "answer": answer}
        for answer in ("red", "blue", "green", "yellow")
    ] + [
        {"split": "train", "question": "How many are there?", "answer": answer}
        for answer in ("1", "2", "3", "4")
    ]
    exact, fallback = compatible_answer_pools(
        rows, split_key="split", question_key="question", answer_key="answer"
    )
    pool = compatible_answer_pool(
        exact,
        fallback,
        split="train",
        question="What color is it?",
        answer="red",
    )
    assert set(pool) == {"red", "blue", "green", "yellow"}
    assert {_answer_kind(value) for value in pool} == {"color"}


def test_open_qa_drops_question_without_compatible_negative() -> None:
    sample = open_qa_sample(
        source="test",
        source_version="test",
        split="train",
        sample_id="test:1",
        group_id="test:1",
        image=None,
        question="What is shown?",
        answer="unique",
        answer_pool=["unique"],
        license_name="test",
        evidence_reference="test",
    )
    assert sample is None


def test_nested_targets_choose_smallest_region() -> None:
    candidates = [
        {"id": "outer", "box": [0, 0, 100, 100]},
        {"id": "inner", "box": [10, 10, 30, 30]},
    ]
    assert _most_specific_target(candidates, ["outer", "inner"]) == "inner"
