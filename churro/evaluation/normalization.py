import re
import unicodedata


# TODO add more
subs = {
    "\ueada": "st",
    "\ueec5": "ct",
    "\ueba6": "ss",
    "\ueba2": "si",
    "\ueba7": "ssi",
    "\ueba3": "sl",
    "’": "'",
    "¬": "-",
}

pattern = re.compile("|".join(map(re.escape, subs.keys())))


def normalize_characters(text: str, keep_long_s: bool = True) -> str:
    """Replace special characters with ASCII equivalents.

    Inserts spaces before certain fractions, preserves long ſ if requested, and applies
    predefined glyph substitutions.
    """
    # Insert a space before common fraction characters when immediately preceded by a digit.
    text = re.sub(r"(?<=\d)(?=[↉½⅓¼⅕⅙⅐⅛⅑⅒⅔⅖¾⅗⅜⅘⅚⅞])", " ", text)

    # Placeholder for long s swap (always define to satisfy type checker)
    placeholder = "\ue000"
    if keep_long_s:
        # Use a placeholder unlikely to appear in normal text.
        text = text.replace("ſ", placeholder)

    # Normalize text to convert special fraction characters (e.g., '½') into their ASCII representations (e.g., '1/2')
    text = unicodedata.normalize("NFKC", text)

    if keep_long_s:
        text = text.replace(placeholder, "ſ")

    # Apply additional substitutions defined elsewhere (subs and pattern should be defined in the module)
    text = pattern.sub(lambda m: subs[m.group(0)], text)

    # Make this change to make the text stay close to the image
    # replace = at the end of a line with a ⸗
    # This is an example of https://en.wikipedia.org/wiki/Double_hyphen
    # This is used in for example newseye_finnish to indicate that a word was split across lines.
    # text = re.sub(r"=(?=\n|$)", "⸗", text)

    # Remove ~ at the beginning of a line. It is used in for example clarysse.
    text = re.sub(r"(^|\s)~(?=\w)", r"\1", text)

    return text


def normalize_text_for_evaluation(
    text: str,
    normalize_arabic: bool = False,
) -> str:
    """Normalize raw OCR/LLM text for evaluation.

    Applies case-folding, punctuation & markdown cleanup, dash standardization,
    removal of figure markers, whitespace collapsing, hyphen-merge, optional
    Arabic normalization, and character substitutions.
    """
    if normalize_arabic:
        from pyarabic.araby import (
            normalize_hamza,
            strip_harakat,
            strip_lastharaka,
            strip_tashkeel,
            strip_tatweel,
        )

        text = strip_tashkeel(text)
        text = strip_harakat(text)
        text = strip_lastharaka(text)
        text = strip_tatweel(text)
        text = normalize_hamza(text)

    # Convert to lowercase.
    text = text.lower()

    # Remove markdown symbols and standardize dashes.
    text = re.sub(r"[*_`~#]", "", text)
    text = re.sub(r"[–—−‑‒―‐]", "-", text)

    # Remove Markdown image blocks.
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)

    # Remove lines that start with '[' and end with ']'. These are often what LLMs output as extra explanations.
    text = re.sub(r"^\s*\[.*\]\s*$", "", text, flags=re.MULTILINE)

    # Remove figure markers like "[figure 1]".
    text = re.sub(r"\[figure\s+\d+\]", "", text)

    # Remove blockquote markers at the start of lines.
    text = re.sub(r"^>\s+", "", text, flags=re.MULTILINE)

    # Remove sequences of three or more hyphens.
    text = re.sub(r"-{3,}", "", text)

    # Remove spaces before punctuation.
    text = re.sub(r"\s+([.,?!;:])", r"\1", text)

    # Merge words split by newline hyphenation.
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
    text = text.strip("-")

    text = normalize_characters(text, keep_long_s=False)

    # Collapse multiple whitespace characters into a single space.
    text = re.sub(r"\s+", " ", text).strip()

    return text


def remove_transcription_tags(text: str) -> str:
    """Remove specialized editorial/marker tags and abbreviation constructs."""
    # Step 1: Convert abbreviated expressions like "Adm.$^r$.Administrador" to "Admr"
    # Pattern:
    #   Capture the base abbreviation, a literal period, a marker ($^letters$), another literal period,
    #   then the full word (which we ignore)
    # print("input text:", text)
    pattern_abbrev = r"\b([A-Za-z]+)\.\$\^([A-Za-z]+)\$\.[A-Za-z]+"
    text = re.sub(pattern_abbrev, r"\1\2", text)

    # Step 2: Convert expressions like "dho$.dicho" to "dho"
    # Pattern explanation:
    #   \b([A-Za-z]+)    : capture a word before the marker (e.g., "dho")
    #   \$\.             : literal "$." sequence
    #   [A-Za-z]+        : one or more letters (e.g., "dicho")
    pattern_dho = r"\b([A-Za-z]+)\$\.[A-Za-z]+"
    text = re.sub(pattern_dho, r"\1", text)

    # Step 3: Remove tags of the form "$tag:" (e.g. "$ant:", "$ofi:", etc.)
    text = re.sub(r"\$[A-Za-z]+:", "", text)

    # Remove markers enclosed in $ signs, e.g. "$^r$" or "$dho$"
    text = re.sub(r"\$[\^A-Za-z]+\$", "", text)

    # text = text.replace("$-", "-")
    text = text.replace(":$-\n$-", "-\n")
    # print("changed text: ", text)

    return text
