import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
import pytest_asyncio

from magentic.chat_model.anthropic_chat_model import AnthropicChatModel
from magentic.chat_model.base import ChatModel
from magentic.chat_model.litellm_chat_model import LitellmChatModel
from magentic.chat_model.message import UserMessage
from magentic.chat_model.mistral_chat_model import MistralChatModel
from magentic.chat_model.openai_chat_model import OpenaiChatModel
from magentic.streaming import AsyncStreamedStr, StreamedStr


@pytest_asyncio.fixture(params=["openai", "anthropic", "litellm", "mistral"])
async def model(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[ChatModel, list[dict[str, Any]]]]:
    requests: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        if request.url.path.endswith("/messages"):
            events: list[dict[str, Any]] = [
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg-local",
                        "type": "message",
                        "role": "assistant",
                        "model": "local",
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 1, "output_tokens": 0},
                    },
                },
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "Hello"},
                },
                {"type": "content_block_stop", "index": 0},
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 1},
                },
                {"type": "message_stop"},
            ]
            content = "".join(
                f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
                for event in events
            )
        else:
            chunk = {
                "id": "chatcmpl-local",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "local",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "Hello"},
                        "finish_reason": "stop",
                    }
                ],
            }
            content = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n"
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=content
        )

    async def async_respond(
        _transport: httpx.AsyncHTTPTransport, request: httpx.Request
    ) -> httpx.Response:
        return respond(request)

    monkeypatch.setattr(
        httpx.HTTPTransport,
        "handle_request",
        lambda _transport, request: respond(request),
    )
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", async_respond)
    monkeypatch.setenv("OPENAI_API_KEY", "local-only-test")
    model: ChatModel
    if request.param == "anthropic":
        model = AnthropicChatModel("local", api_key="local-only-test")
    elif request.param == "litellm":
        model = LitellmChatModel("openai/local")
    elif request.param == "mistral":
        model = MistralChatModel("local", api_key="local-only-test")
    else:
        model = OpenaiChatModel("local", api_key="local-only-test")
    try:
        yield model, requests
    finally:
        if isinstance(model, MistralChatModel):
            model = model._mistral_openai_chat_model
        if isinstance(model, OpenaiChatModel | AnthropicChatModel):
            model._client.close()
            await model._async_client.close()


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("as_iterator", [False, True])
@pytest.mark.parametrize("kind", ["text", "streamed", "text-or-int"])
async def test_output_types_accept_one_shot_iterables(
    model: tuple[ChatModel, list[dict[str, Any]]],
    *,
    is_async: bool,
    as_iterator: bool,
    kind: str,
) -> None:
    chat_model, requests = model
    output_types: list[type[Any]]
    if kind == "streamed":
        output_types = [AsyncStreamedStr if is_async else StreamedStr]
    elif kind == "text-or-int":
        output_types = [str, int]
    else:
        output_types = [str]
    allowed = iter(output_types) if as_iterator else output_types
    if is_async:
        response = await chat_model.acomplete([UserMessage("Hi")], output_types=allowed)
    else:
        response = chat_model.complete([UserMessage("Hi")], output_types=allowed)

    if kind == "streamed":
        assert isinstance(
            response.content, AsyncStreamedStr if is_async else StreamedStr
        )
        content = (
            await response.content.to_string() if is_async else str(response.content)
        )
    else:
        content = response.content
    assert content == "Hello"
    assert len(requests) == 1
    assert requests[0].get("tool_choice") in (None, "auto")
