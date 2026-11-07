from types import SimpleNamespace

import pytest
import requests
import yaml

from llmtrust.models import (AnthropicClient, Arm, OpenAICompatClient,
                             load_arms, make_client)

ARM_FIELDS = dict(provider="openai_compat", model="m1", base_url="http://h/v1/",
                  api_key_env="FAKE_KEY_ENV", temperature=0.0,
                  usd_per_mtok_in=1.0, usd_per_mtok_out=2.0, pool="other")


def _arm(name="a", **kw):
    f = dict(ARM_FIELDS)
    f.update(kw)
    return Arm(name=name, **f)


class FakeResp:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code = status
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


PAYLOAD = {"model": "server-model-9", "choices": [{"message": {"content": "hi"}}],
           "usage": {"prompt_tokens": 11, "completion_tokens": 3}}


def test_load_arms_validates_schema(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump({"arms": {"x": {**ARM_FIELDS, "provider": "bogus"}}}))
    with pytest.raises(ValueError, match="provider"):
        load_arms(p)
    p.write_text(yaml.safe_dump({"arms": {"x": {**ARM_FIELDS, "pool": "bogus"}}}))
    with pytest.raises(ValueError, match="pool"):
        load_arms(p)


def test_load_arms_allows_null_price_on_paid_arm(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump({"arms": {"x": {**ARM_FIELDS, "usd_per_mtok_in": None,
                                                 "usd_per_mtok_out": None}}}))
    arms = load_arms(p)
    assert arms["x"].name == "x" and arms["x"].usd_per_mtok_in is None


def test_repo_models_yaml_loads():
    from pathlib import Path
    arms = load_arms(Path(__file__).resolve().parent.parent / "config" / "models.yaml")
    assert {"claude", "dev-free", "open-large", "local-small"} <= set(arms)
    assert all(a.temperature == 0.0 for a in arms.values())
    assert arms["local-small"].api_key_env is None
    assert arms["claude"].pool == "claude"


def test_openai_compat_request_shape(monkeypatch):
    monkeypatch.setenv("FAKE_KEY_ENV", "sekret")
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append((url, headers, json))
        return FakeResp(200, PAYLOAD)

    monkeypatch.setattr(requests, "post", fake_post)
    c = make_client(_arm())
    assert isinstance(c, OpenAICompatClient)
    out = c.complete("sys", "usr", max_tokens=50)
    url, headers, body = calls[0]
    assert url == "http://h/v1/chat/completions"
    assert headers["Authorization"] == "Bearer sekret"
    assert body["temperature"] == 0.0
    assert body["max_tokens"] == 50 and body["model"] == "m1"
    assert body["messages"] == [{"role": "system", "content": "sys"},
                                {"role": "user", "content": "usr"}]
    assert out.text == "hi"
    assert out.model_id == "server-model-9"
    assert out.provider == "openai_compat"
    assert (out.input_tokens, out.output_tokens) == (11, 3)
    assert out.finished_at.endswith("+00:00") or out.finished_at.endswith("Z")


def test_openai_compat_omits_temperature_and_auth_when_unset(monkeypatch):
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append((headers, json))
        return FakeResp(200, {**PAYLOAD, "model": None})

    monkeypatch.setattr(requests, "post", fake_post)
    out = make_client(_arm(temperature=None, api_key_env=None)).complete("s", "u", max_tokens=5)
    headers, body = calls[0]
    assert "temperature" not in body
    assert "Authorization" not in headers
    assert out.model_id == "m1"  # falls back to arm.model


def test_openai_compat_error_names_arm_and_status(monkeypatch):
    monkeypatch.setenv("FAKE_KEY_ENV", "k")
    monkeypatch.setattr(requests, "post",
                        lambda *a, **k: FakeResp(429, None, "slow down"))
    with pytest.raises(RuntimeError, match=r"arm-x.*429"):
        make_client(_arm(name="arm-x")).complete("s", "u", max_tokens=5)


def test_openai_compat_uses_injected_session(monkeypatch):
    monkeypatch.setenv("FAKE_KEY_ENV", "k")
    seen = []
    session = SimpleNamespace(post=lambda url, **kw: (seen.append(url), FakeResp(200, PAYLOAD))[1])
    make_client(_arm(base_url="http://h/v1"), session=session).complete("s", "u", max_tokens=5)
    assert seen == ["http://h/v1/chat/completions"]


class FakeMessages:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            model="claude-haiku-5-5-20260101",
            content=[SimpleNamespace(type="text", text="ans")],
            usage=SimpleNamespace(input_tokens=20, output_tokens=4))


def _anthropic_arm(**kw):
    f = dict(provider="anthropic", base_url=None, api_key_env="ANTHROPIC_API_KEY",
             model="claude-haiku-5-5", pool="claude")
    f.update(kw)
    return _arm(**f)


def test_anthropic_client_request_and_completion():
    fake = SimpleNamespace(messages=FakeMessages())
    c = make_client(_anthropic_arm(), sdk_client=fake)
    assert isinstance(c, AnthropicClient)
    out = c.complete("sys", "usr", max_tokens=99)
    kw = fake.messages.kwargs
    assert kw["model"] == "claude-haiku-5-5" and kw["max_tokens"] == 99
    assert kw["system"] == "sys" and kw["temperature"] == 0.0
    assert kw["messages"] == [{"role": "user", "content": "usr"}]
    assert out.text == "ans" and out.model_id == "claude-haiku-5-5-20260101"
    assert out.provider == "anthropic"
    assert (out.input_tokens, out.output_tokens) == (20, 4)


def test_anthropic_client_omits_temperature_when_null():
    fake = SimpleNamespace(messages=FakeMessages())
    make_client(_anthropic_arm(temperature=None), sdk_client=fake).complete("s", "u", max_tokens=5)
    assert "temperature" not in fake.messages.kwargs


def test_unknown_provider_in_make_client():
    with pytest.raises(ValueError):
        make_client(_arm(provider="nope"))


class FakeStatusError(Exception):
    def __init__(self, status_code, message="boom"):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def _raising_sdk(exc):
    def create(**kwargs):
        raise exc
    return SimpleNamespace(messages=SimpleNamespace(create=create))


def test_anthropic_status_error_becomes_runtime_error_naming_arm_and_status():
    sdk = _raising_sdk(FakeStatusError(529, "overloaded"))
    c = make_client(_anthropic_arm(name="claude-x"), sdk_client=sdk)
    with pytest.raises(RuntimeError, match=r"claude-x.*529") as ei:
        c.complete("s", "u", max_tokens=5)
    assert "overloaded" in str(ei.value)


def test_anthropic_real_sdk_status_error_is_wrapped():
    import anthropic
    import httpx
    resp = httpx.Response(429, request=httpx.Request("POST", "http://offline.invalid"))
    exc = anthropic.RateLimitError("rate limited", response=resp, body={"secret": "k-123"})
    c = make_client(_anthropic_arm(name="claude-y"), sdk_client=_raising_sdk(exc))
    with pytest.raises(RuntimeError, match=r"claude-y.*429") as ei:
        c.complete("s", "u", max_tokens=5)
    assert "k-123" not in str(ei.value)


def test_anthropic_non_status_error_propagates_unchanged():
    c = make_client(_anthropic_arm(), sdk_client=_raising_sdk(ValueError("bad")))
    with pytest.raises(ValueError):
        c.complete("s", "u", max_tokens=5)


def test_anthropic_sdk_client_built_with_no_retries(monkeypatch):
    import anthropic
    captured = {}

    class FakeAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.messages = FakeMessages()

    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-not-real")
    make_client(_anthropic_arm()).complete("s", "u", max_tokens=5)
    assert captured["max_retries"] == 0
    assert captured["api_key"] == "dummy-not-real"
