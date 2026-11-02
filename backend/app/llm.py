import random
import time
from dataclasses import dataclass

from .policy import estimate_tokens, llm_cost


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


class LLMProvider:
    def __init__(self, settings):
        self.s = settings

    def _complete(self, prompt: str, max_tokens: int) -> tuple[str, int, int]:
        raise NotImplementedError

    def generate(self, query: str) -> LLMResult:
        text, tin, tout = self._complete(query, self.s.llm_max_tokens)
        return LLMResult(text, tin, tout, llm_cost(tin, tout, self.s.price_input_per_m, self.s.price_output_per_m))

    def judge_same(self, a: str, b: str) -> bool:
        prompt = (
            "Do these two questions ask for exactly the same thing, so that one answer would be "
            "correct for both?\n"
            f"Question A: {a}\nQuestion B: {b}\nReply with only YES or NO."
        )
        text, _, _ = self._complete(prompt, 5)
        return text.strip().upper().startswith("YES")


class MockProvider(LLMProvider):
    """Simulates an LLM: fixed-ish latency, token-based cost. No API key needed."""

    def _complete(self, prompt, max_tokens):
        base = self.s.mock_latency_ms
        time.sleep(max(0.0, base + random.uniform(-0.2, 0.2) * base) / 1000)
        text = f"(mock answer) You asked: {prompt}"
        return text, estimate_tokens(prompt), 80 + len(prompt) // 2

    def judge_same(self, a, b):
        ta, tb = set(a.lower().split()), set(b.lower().split())
        return len(ta & tb) / max(1, len(ta | tb)) >= 0.5


class AnthropicProvider(LLMProvider):
    def __init__(self, settings):
        super().__init__(settings)
        import anthropic

        self.client = anthropic.Anthropic()

    def _complete(self, prompt, max_tokens):
        r = self.client.messages.create(
            model=self.s.llm_model, max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}]
        )
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        return text, r.usage.input_tokens, r.usage.output_tokens


class OpenAIProvider(LLMProvider):
    def __init__(self, settings):
        super().__init__(settings)
        import openai

        self.client = openai.OpenAI()

    def _complete(self, prompt, max_tokens):
        r = self.client.chat.completions.create(
            model=self.s.llm_model, max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}]
        )
        return r.choices[0].message.content or "", r.usage.prompt_tokens, r.usage.completion_tokens


def get_llm(settings) -> LLMProvider:
    p = settings.llm_provider
    if p == "anthropic":
        return AnthropicProvider(settings)
    if p == "openai":
        return OpenAIProvider(settings)
    return MockProvider(settings)
