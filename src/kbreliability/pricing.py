"""Illustrative token pricing for the cost/latency arbitrage line.

USD per 1M tokens (input / output), public list prices, order-of-magnitude
only. Unknown model -> no dollar estimate.
"""

from __future__ import annotations

_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.4, 1.6),
}


def model_from_name(name: str) -> str | None:
    """Extract a model from a system name like 'llm:gpt-4o' (else None)."""
    _, sep, model = name.partition(":")
    return model if sep and model else None


def estimate_usd(model: str | None, prompt_tokens: int, completion_tokens: int) -> float | None:
    if model is None:
        return None
    price = _PRICES.get(model) or _PRICES.get(model.removeprefix("openai/"))
    if price is None:
        return None
    input_per_m, output_per_m = price
    return prompt_tokens / 1_000_000 * input_per_m + completion_tokens / 1_000_000 * output_per_m
