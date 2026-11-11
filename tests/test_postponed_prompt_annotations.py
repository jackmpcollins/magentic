from __future__ import annotations

import json
from functools import partial
from typing import Any, TypeAlias

import httpx
import openai
import pytest
import pytest_asyncio

from magentic import UserMessage, chatprompt, prompt, prompt_chain
from magentic.chat_model.openai_chat_model import OpenaiChatModel

Reply: TypeAlias = str


@pytest_asyncio.fixture
async def model(monkeypatch):
    def respond(request):
        body = json.loads(request.content)
        chunk = {
            "id": "chatcmpl-local",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "local",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": body["messages"][-1]["content"],
                    },
                    "finish_reason": "stop",
                }
            ],
        }
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
        )

    transport = httpx.MockTransport(respond)
    with httpx.Client(transport=transport) as client:
        async with httpx.AsyncClient(transport=transport) as async_client:
            monkeypatch.setattr(
                openai, "OpenAI", partial(openai.OpenAI, http_client=client)
            )
            monkeypatch.setattr(
                openai,
                "AsyncOpenAI",
                partial(openai.AsyncOpenAI, http_client=async_client),
            )
            yield OpenaiChatModel("local", api_key="local-only-test")


@pytest.mark.parametrize("kind", ["prompt", "chatprompt", "prompt_chain"])
@pytest.mark.parametrize("is_async", [False, True])
async def test_postponed_prompt_annotations(model, kind, is_async):
    decorate: Any
    if kind == "chatprompt":
        decorate = chatprompt(UserMessage("Hello {name}"), model=model)
    elif kind == "prompt_chain":
        decorate = prompt_chain("Hello {name}", model=model)
    else:
        decorate = prompt("Hello {name}", model=model)

    if is_async:

        async def agreeting(name: str = "Ada") -> Reply: ...

        assert await decorate(agreeting)() == "Hello Ada"
    else:

        def greeting(name: str = "Ada") -> Reply: ...

        assert decorate(greeting)() == "Hello Ada"
