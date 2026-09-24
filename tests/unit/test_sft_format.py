from __future__ import annotations

from vision_jev.train.sft import answer_text, question_text


def test_sft_choice_format_uses_candidate_id() -> None:
    sample = {
        "task_type": "choice",
        "target": "option_1",
        "question": "Pick one",
        "state_text": "",
        "options": [
            {"id": "option_0", "text": "left"},
            {"id": "option_1", "text": "right"},
        ],
    }
    assert answer_text(sample) == '{"choice":"option_1"}'
    prompt = question_text(sample)
    assert "option_0: left" in prompt
    assert "option_1: right" in prompt


def test_sft_noul_format_uses_json_boolean() -> None:
    sample = {
        "task_type": "noul",
        "target": False,
        "question": "Is it supported?",
        "state_text": "visible",
        "options": [],
    }
    assert answer_text(sample) == '{"noul":false}'
    assert "Visible state: visible" in question_text(sample)
