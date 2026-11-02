"""Pure decision logic (no I/O) so it is trivial to unit test."""
import re


def decide(sim: float, threshold: float, mode: str, low: float, high: float) -> str:
    """Return 'hit', 'verify' or 'miss' for a similarity score.

    - mode 'none': one threshold.
    - otherwise: >= high is an automatic hit, [low, high) goes to the
      second-stage verifier, < low is a miss.
    """
    if mode == "none":
        return "hit" if sim >= threshold else "miss"
    if sim >= high:
        return "hit"
    if sim >= low:
        return "verify"
    return "miss"


def is_time_sensitive(query: str, pattern: str) -> bool:
    return bool(pattern) and re.search(pattern, query, re.IGNORECASE) is not None


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def llm_cost(input_tokens: int, output_tokens: int, price_in_per_m: float, price_out_per_m: float) -> float:
    return (input_tokens * price_in_per_m + output_tokens * price_out_per_m) / 1_000_000
