from app.policy import decide, is_time_sensitive, llm_cost
from app.config import settings


def test_single_threshold():
    assert decide(0.91, 0.90, "none", 0.8, 0.93) == "hit"
    assert decide(0.89, 0.90, "none", 0.8, 0.93) == "miss"


def test_two_stage_bands():
    assert decide(0.95, 0.90, "crossencoder", 0.8, 0.93) == "hit"
    assert decide(0.85, 0.90, "crossencoder", 0.8, 0.93) == "verify"
    assert decide(0.70, 0.90, "crossencoder", 0.8, 0.93) == "miss"


def test_time_sensitive():
    p = settings.time_sensitive_regex
    assert is_time_sensitive("What's the weather in Paris today?", p)
    assert is_time_sensitive("latest news on AI", p)
    assert not is_time_sensitive("How do I reset my password?", p)


def test_cost():
    assert abs(llm_cost(1_000_000, 1_000_000, 1.0, 5.0) - 6.0) < 1e-9
