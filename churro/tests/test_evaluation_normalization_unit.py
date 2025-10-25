from __future__ import annotations

from churro.evaluation import normalization


def test_normalize_characters_preserves_long_s_when_requested() -> None:
    text = "ſale 1½"
    result = normalization.normalize_characters(text, keep_long_s=True)
    assert "ſ" in result
    assert "1 1⁄2" in result


def test_normalize_characters_replaces_long_s_when_disabled() -> None:
    text = "ſafety"
    result = normalization.normalize_characters(text, keep_long_s=False)
    assert "ſ" not in result
    assert "safety" in result


def test_normalize_text_for_evaluation_strips_markdown_and_hyphenation() -> None:
    raw = "# Heading\n> Quote line\npara-\ngraph ![img](link)\n[figure 2]\nword  —  spaced\n"
    result = normalization.normalize_text_for_evaluation(raw)
    assert "![img]" not in result
    assert "paragraph" in result
    assert "para-\n" not in result
    assert "word - spaced" in result


def test_normalize_text_for_evaluation_with_arabic_normalization() -> None:
    raw = "السَّلَامُ"
    result = normalization.normalize_text_for_evaluation(raw, normalize_arabic=True)
    assert "َ" not in result  # tashkeel removed


def test_remove_transcription_tags() -> None:
    raw = "Adm.$^r$.Administrador dho$.dicho $ant: text $dho$"
    result = normalization.remove_transcription_tags(raw)
    assert result == "Admr dho  text "
