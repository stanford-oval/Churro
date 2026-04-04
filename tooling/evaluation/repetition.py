"""Repetition heuristics used by repo-only evaluation tooling."""

from __future__ import annotations


def has_long_repetition(text: str) -> bool:
    """Return True when the tail of the string is composed of repeated content."""
    length = len(text)
    if length < 2:
        return False

    reversed_text = text[::-1]
    prefix_function = [0] * length
    for i in range(1, length):
        j = prefix_function[i - 1]
        while j and reversed_text[i] != reversed_text[j]:
            j = prefix_function[j - 1]
        if reversed_text[i] == reversed_text[j]:
            j += 1
        prefix_function[i] = j

    max_prefix = int(0.8 * length)
    for prefix_size in range(1, max_prefix + 1):
        remainder = length - prefix_size
        if remainder < 2:
            continue
        border = prefix_function[remainder - 1]
        period = remainder - border
        if border > 0 and remainder % period == 0 and remainder // period >= 2:
            return True

    return False
