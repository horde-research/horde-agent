from __future__ import annotations

import httpx

from core.llm.client import LLMClient, LLMRequest


class FakeAsyncClient:
    def __init__(self, *args, **kwargs) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def post(self, url, *, json=None, params=None, headers=None):
        request = httpx.Request("POST", httpx.URL(url, params=params))
        return httpx.Response(403, request=request, text='{"error":"denied"}')


def test_gemini_error_redacts_api_key_from_response(monkeypatch) -> None:
    monkeypatch.setattr("core.llm.client.httpx.AsyncClient", FakeAsyncClient)
    client = LLMClient("gemini", "gemini-test", "gemini-secret", max_retries=1)

    response = client.generate_json_sync(LLMRequest(request_id="req-1", user_message="hello"))

    assert not response.success
    assert response.error
    assert "gemini-secret" not in response.error
    assert "key=[REDACTED]" in response.error
