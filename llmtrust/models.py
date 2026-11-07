from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import requests
import yaml

from llmtrust.types import Completion

PROVIDERS = ("anthropic", "openai_compat")
POOLS = ("claude", "other", "free")
REQUEST_TIMEOUT_S = 300


@dataclass(frozen=True)
class Arm:
    name: str
    provider: str
    model: str
    base_url: str | None
    api_key_env: str | None
    temperature: float | None
    usd_per_mtok_in: float | None
    usd_per_mtok_out: float | None
    pool: str


def load_arms(path: Path) -> dict[str, Arm]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    arms: dict[str, Arm] = {}
    for name, f in (raw.get("arms") or {}).items():
        if f.get("provider") not in PROVIDERS:
            raise ValueError(f"arm {name!r}: unknown provider {f.get('provider')!r}; "
                             f"expected one of {PROVIDERS}")
        if f.get("pool") not in POOLS:
            raise ValueError(f"arm {name!r}: unknown pool {f.get('pool')!r}; "
                             f"expected one of {POOLS}")
        temp = f.get("temperature")
        arms[name] = Arm(
            name=name,
            provider=f["provider"],
            model=f["model"],
            base_url=f.get("base_url"),
            api_key_env=f.get("api_key_env"),
            temperature=None if temp is None else float(temp),
            usd_per_mtok_in=f.get("usd_per_mtok_in"),
            usd_per_mtok_out=f.get("usd_per_mtok_out"),
            pool=f["pool"],
        )
    return arms


class ModelClient(Protocol):
    def complete(self, system: str, user: str, *, max_tokens: int) -> Completion: ...


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class AnthropicClient:
    def __init__(self, arm: Arm, sdk_client: Any = None):
        self.arm = arm
        if sdk_client is None:
            import anthropic
            key = os.environ.get(arm.api_key_env) if arm.api_key_env else None
            sdk_client = anthropic.Anthropic(api_key=key, max_retries=0)
        self._sdk = sdk_client

    def complete(self, system: str, user: str, *, max_tokens: int) -> Completion:
        kwargs: dict[str, Any] = dict(
            model=self.arm.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if self.arm.temperature is not None:
            kwargs["temperature"] = self.arm.temperature
        try:
            resp = self._sdk.messages.create(**kwargs)
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if status is None:
                raise
            short = str(getattr(exc, "message", "") or "")[:200]
            raise RuntimeError(f"arm {self.arm.name!r}: HTTP {status} from anthropic: {short}") from exc
        text = "".join(b.text for b in resp.content if getattr(b, "type", "text") == "text")
        return Completion(
            text=text,
            model_id=getattr(resp, "model", None) or self.arm.model,
            provider="anthropic",
            input_tokens=int(resp.usage.input_tokens),
            output_tokens=int(resp.usage.output_tokens),
            finished_at=_now_iso(),
        )


class OpenAICompatClient:
    def __init__(self, arm: Arm, session: Any = None):
        if not arm.base_url:
            raise ValueError(f"arm {arm.name!r}: openai_compat needs a base_url")
        self.arm = arm
        self._post = session.post if session is not None else None

    def complete(self, system: str, user: str, *, max_tokens: int) -> Completion:
        arm = self.arm
        headers = {"Content-Type": "application/json"}
        if arm.api_key_env:
            headers["Authorization"] = f"Bearer {os.environ.get(arm.api_key_env, '')}"
        body: dict[str, Any] = {
            "model": arm.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        }
        if arm.temperature is not None:
            body["temperature"] = arm.temperature
        post = self._post or requests.post
        resp = post(arm.base_url.rstrip("/") + "/chat/completions",
                    headers=headers, json=body, timeout=REQUEST_TIMEOUT_S)
        if not 200 <= resp.status_code < 300:
            raise RuntimeError(f"arm {arm.name!r}: HTTP {resp.status_code} from "
                               f"{arm.base_url}: {str(resp.text)[:300]}")
        data = resp.json()
        usage = data.get("usage") or {}
        return Completion(
            text=data["choices"][0]["message"]["content"] or "",
            model_id=data.get("model") or arm.model,
            provider="openai_compat",
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            finished_at=_now_iso(),
        )


def make_client(arm: Arm, *, sdk_client: Any = None, session: Any = None) -> ModelClient:
    if arm.provider == "anthropic":
        return AnthropicClient(arm, sdk_client=sdk_client)
    if arm.provider == "openai_compat":
        return OpenAICompatClient(arm, session=session)
    raise ValueError(f"arm {arm.name!r}: unknown provider {arm.provider!r}")
