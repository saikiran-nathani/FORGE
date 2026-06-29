"""Unified LLM client with OpenAI support and heuristic fallback."""

from __future__ import annotations

import json
import os
import re
from typing import Any


class LLMClient:
    """Calls OpenAI (or Anthropic) for JSON/text completions; falls back to heuristics."""

    def __init__(
        self,
        provider: str = "openai",
        model: str | None = None,
        api_key: str | None = None,
    ):
        self.provider = provider
        self.model = model or os.getenv("FORGE_LLM_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        if not self.available:
            raise RuntimeError("No LLM API key configured")
        if self.provider == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
            return self._anthropic_complete(system, user)
        return self._openai_complete(system, user, json_mode)

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        if not self.available:
            return {}
        text = self.complete(system, user, json_mode=True)
        return self._parse_json(text)

    def _openai_complete(self, system: str, user: str, json_mode: bool) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def _anthropic_complete(self, system: str, user: str) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model=self.model if "claude" in self.model else "claude-3-5-haiku-20241022",
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
        return json.loads(text)
