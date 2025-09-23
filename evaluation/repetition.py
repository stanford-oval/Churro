def has_long_repetition(s: str) -> bool:
    """Return True iff s = A + B*k (k>=2) with nonempty A,B and len(A) <= 0.8*len(s)."""
    n: int = len(s)
    if n < 2:
        return False

    # compute prefix‐function on reversed string
    r: str = s[::-1]
    pi: list[int] = [0] * n
    for i in range(1, n):
        j: int = pi[i - 1]
        while j and r[i] != r[j]:
            j = pi[j - 1]
        if r[i] == r[j]:
            j += 1
        pi[i] = j

    max_a: int = int(0.8 * n)
    for a in range(1, max_a + 1):
        rem: int = n - a
        if rem < 2:
            continue
        border: int = pi[rem - 1]
        p: int = rem - border
        # B must repeat at least twice
        if border > 0 and rem % p == 0 and rem // p >= 2:
            return True

    return False


# def remove_repetition(text: str) -> str:
#     """
#     Remove long repetitions from the text.
#     """
#     if not text:
#         return text
#     result, count = longest_repeated_substring_nonoverlapping(text, 10)
#     ratio = len(result) * count / len(text)
#     if ratio > 0.1:
#         return text.replace(result, "")
#     return text
