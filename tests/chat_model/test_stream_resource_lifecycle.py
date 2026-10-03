import json

import httpx
import openai
import pytest

from magentic import (
    AsyncStreamedResponse,
    AsyncStreamedStr,
    StreamedResponse,
    StreamedStr,
)
from magentic.chat_model.message import UserMessage
from magentic.chat_model.openai_chat_model import OpenaiChatModel


def chunks():
    for text in ("Hello", " world"):
        yield (
            "data: "
            + json.dumps(
                {
                    "id": "test",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "test",
                    "choices": [
                        {"index": 0, "delta": {"content": text}, "finish_reason": None}
                    ],
                }
            )
            + "\n\n"
        ).encode()
    yield b"data: [DONE]\n\n"


class TrackingStream(httpx.SyncByteStream):
    def __init__(self):
        self.close_calls = 0

    def __iter__(self):
        yield from chunks()

    def close(self):
        self.close_calls += 1


class AsyncTrackingStream(httpx.AsyncByteStream):
    def __init__(self):
        self.close_calls = 0

    async def __aiter__(self):
        for chunk in chunks():
            yield chunk

    async def aclose(self):
        self.close_calls += 1


@pytest.mark.parametrize("output_type", [StreamedStr, StreamedResponse])
def test_context_closes_openai_response_after_early_stop(output_type):
    body = TrackingStream()
    transport = httpx.MockTransport(lambda _: httpx.Response(200, stream=body))
    model = OpenaiChatModel("test", api_key="test")
    model._client = openai.OpenAI(
        api_key="test", http_client=httpx.Client(transport=transport)
    )
    response = model.complete([UserMessage("hi")], output_types=[output_type]).content
    with response:
        text = response if isinstance(response, StreamedStr) else next(iter(response))
        assert next(iter(text)) == "Hello"
        assert body.close_calls == 0
    response.close()
    assert body.close_calls == 1
    assert str(text) == "Hello"


@pytest.mark.asyncio
@pytest.mark.parametrize("output_type", [AsyncStreamedStr, AsyncStreamedResponse])
async def test_async_context_closes_openai_response_after_early_stop(output_type):
    body = AsyncTrackingStream()
    transport = httpx.MockTransport(lambda _: httpx.Response(200, stream=body))
    model = OpenaiChatModel("test", api_key="test")
    model._async_client = openai.AsyncOpenAI(
        api_key="test", http_client=httpx.AsyncClient(transport=transport)
    )
    response = (
        await model.acomplete([UserMessage("hi")], output_types=[output_type])
    ).content
    async with response:
        text = (
            response
            if isinstance(response, AsyncStreamedStr)
            else await anext(aiter(response))
        )
        assert await anext(aiter(text)) == "Hello"
        assert body.close_calls == 0
    await response.aclose()
    assert body.close_calls == 1
    assert await text.to_string() == "Hello"


def test_exhausted_text_closes_response_and_replays_cache():
    body = TrackingStream()
    model = OpenaiChatModel("test", api_key="test")
    model._client = openai.OpenAI(
        api_key="test",
        http_client=httpx.Client(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=body))
        ),
    )
    text = model.complete([UserMessage("hi")], output_types=[StreamedStr]).content
    assert str(text) == "Hello world"
    assert body.close_calls == 1
    assert str(text) == "Hello world"


@pytest.mark.parametrize("output_type", [StreamedStr, StreamedResponse])
def test_producer_error_closes_response(output_type):
    class FailedStream(TrackingStream):
        def __iter__(self):
            yield next(chunks())
            raise RuntimeError

    body = FailedStream()
    model = OpenaiChatModel("test", api_key="test")
    model._client = openai.OpenAI(
        api_key="test",
        http_client=httpx.Client(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=body))
        ),
    )
    response = model.complete([UserMessage("hi")], output_types=[output_type]).content
    text = response if isinstance(response, StreamedStr) else next(iter(response))
    with pytest.raises(RuntimeError):
        str(text)
    assert body.close_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("output_type", [AsyncStreamedStr, AsyncStreamedResponse])
async def test_consumer_cancellation_closes_response(output_type):
    import asyncio

    entered = asyncio.Event()

    class BlockingStream(AsyncTrackingStream):
        async def __aiter__(self):
            yield next(chunks())
            entered.set()
            await asyncio.Event().wait()

    body = BlockingStream()
    model = OpenaiChatModel("test", api_key="test")
    model._async_client = openai.AsyncOpenAI(
        api_key="test",
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=body))
        ),
    )
    response = (
        await model.acomplete([UserMessage("hi")], output_types=[output_type])
    ).content
    text = (
        response
        if isinstance(response, AsyncStreamedStr)
        else await anext(aiter(response))
    )
    task = asyncio.create_task(text.to_string())
    await asyncio.wait_for(entered.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert body.close_calls == 1
