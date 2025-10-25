"""Lightweight helpers for parsing and conversions used with LLM outputs."""

from churro.utils.log_utils import logger


def extract_tag_from_llm_output(llm_output: str, tags: str | list[str]) -> str | list[str]:
    """Extract tagged content from churro.utils.llm output."""
    is_list = isinstance(tags, list)
    if not is_list:
        assert isinstance(tags, str)
        tags = [tags]
    all_extracted_tags: list[str] = []
    for tag in tags:
        extracted_tag = ""
        open_tag = f"<{tag}>"
        close_tag = f"</{tag}>"
        start_idx = llm_output.find(open_tag)
        if start_idx == -1:
            all_extracted_tags.append("")
            continue
        content_start = start_idx + len(open_tag)
        end_idx = llm_output.find(close_tag, content_start)
        if end_idx == -1:
            all_extracted_tags.append("")
            continue
        extracted_tag = llm_output[content_start:end_idx].strip()
        all_extracted_tags.append(extracted_tag)

    if not is_list:
        return all_extracted_tags[0]
    return all_extracted_tags


def string_to_list_of_floats(string: str) -> list[float]:
    """Parse a string representation of a float list.

    Example input: "[0.5, 0.6, 0.7, 45]"
    """
    try:
        string = string.strip("[]").strip()
        if not string:
            return []
        return [float(x.strip()) for x in string.split(",")]
    except Exception as e:
        logger.warning(f"Failed to parse float list from '{string}': {e}")
        return []


def string_to_list_of_ints(string: str) -> list[int]:
    """Parse a string representation of an int list.

    Example input: "[1, 2, 3, 4]"
    """
    try:
        string = string.strip("[]").strip()
        if not string:
            return []
        return [int(x.strip()) for x in string.split(",")]
    except Exception as e:
        logger.warning(f"Failed to parse int list from '{string}': {e}")
        return []
