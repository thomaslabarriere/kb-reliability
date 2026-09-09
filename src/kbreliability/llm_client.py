"""Shared OpenAI-compatible client construction (openai / openrouter).

One typed, public helper so the answerer, judge, and reranker build their
client the same way instead of reaching into each other's private functions.
The `openai` import stays lazy so the offline paths never import the SDK.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def make_client(
    provider: str = "openai", api_key: str | None = None, base_url: str | None = None
) -> OpenAI:
    from openai import OpenAI

    if base_url is None and provider == "openrouter":
        base_url = _OPENROUTER_BASE_URL
    if api_key is None:
        api_key = os.environ.get(
            "OPENROUTER_API_KEY" if provider == "openrouter" else "OPENAI_API_KEY"
        )
    return OpenAI(api_key=api_key, base_url=base_url)
