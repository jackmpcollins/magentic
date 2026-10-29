import json
from typing import Any

import httpx
import openai
import pytest
from pydantic import BaseModel

from magentic.chat_model.message import Usage, UserMessage
from magentic.chat_model.openai_chat_model import OpenaiChatModel
from magentic.function_call import FunctionCall, ParallelFunctionCall
from magentic.streaming import StreamedStr


def completion(message: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "compatible-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", **message},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11},
    }


def model_for(
    payload: dict[str, Any], requests: list[dict[str, Any]]
) -> OpenaiChatModel:
    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        assert body["stream"] is False
        assert "stream_options" not in body
        return httpx.Response(200, json=payload)

    model = OpenaiChatModel("compatible-model", api_key="test", stream=False)
    model._client = openai.OpenAI(
        api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handle))
    )
    model._async_client = openai.AsyncOpenAI(
        api_key="test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )
    return model


def test_nonstream_text():
    requests: list[dict[str, Any]] = []
    model = model_for(completion({"content": "hello"}), requests)
    result = model.complete([UserMessage("Hi")], stop=["END"])
    assert result.content == "hello"
    assert result.usage == Usage(7, 4)
    assert requests[0]["stop"] == ["END"]


async def test_async_nonstream_text():
    model = model_for(completion({"content": "hello"}), [])
    result = await model.acomplete([UserMessage("Hi")])
    assert result.content == "hello"
    assert result.usage == Usage(7, 4)


def plus(a: int, b: int) -> int:
    return a + b


def test_nonstream_function():
    tool = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "plus", "arguments": '{"a":2,"b":3}'},
    }
    model = model_for(completion({"content": None, "tool_calls": [tool]}), [])
    result = model.complete(
        [UserMessage("Add")], functions=[plus], output_types=[FunctionCall[int]]
    )
    assert result.content() == 5
    assert result.usage == Usage(7, 4)


def test_nonstream_lazy_text():
    model = model_for(completion({"content": "hello"}), [])
    result = model.complete([UserMessage("Hi")], output_types=[StreamedStr])
    assert str(result.content) == "hello"
    assert result.usage == Usage(7, 4)


class Result(BaseModel):
    value: int


def test_nonstream_structured_output():
    tool = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "return_result", "arguments": '{"value":5}'},
    }
    requests: list[dict[str, Any]] = []
    model = model_for(completion({"content": None, "tool_calls": [tool]}), requests)
    result = model.complete([UserMessage("Value")], output_types=[Result])
    assert result.content == Result(value=5)
    assert result.usage == Usage(7, 4)


def test_nonstream_parallel_functions():
    tools = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "plus", "arguments": '{"a":2,"b":3}'},
        },
        {
            "id": "call_2",
            "type": "function",
            "function": {"name": "plus", "arguments": '{"a":4,"b":5}'},
        },
    ]
    model = model_for(completion({"content": None, "tool_calls": tools}), [])
    result = model.complete(
        [UserMessage("Add")], functions=[plus], output_types=[ParallelFunctionCall[int]]
    )
    assert result.content() == (5, 9)
    assert result.usage == Usage(7, 4)


async def test_async_nonstream_function():
    tool = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "plus", "arguments": '{"a":2,"b":3}'},
    }
    model = model_for(completion({"content": None, "tool_calls": [tool]}), [])
    result = await model.acomplete(
        [UserMessage("Add")], functions=[plus], output_types=[FunctionCall[int]]
    )
    assert result.content() == 5
    assert result.usage == Usage(7, 4)


def test_default_stream_request():
    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["stream"] is True
        assert body["stream_options"] == {"include_usage": True}
        chunks = [
            {
                "id": "x",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "test",
                "choices": [
                    {"index": 0, "delta": {"content": "hi"}, "finish_reason": None}
                ],
            },
            {
                "id": "x",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "test",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": 4,
                    "total_tokens": 11,
                },
            },
        ]
        content = (
            "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks)
            + "data: [DONE]\n\n"
        )
        return httpx.Response(
            200, text=content, headers={"content-type": "text/event-stream"}
        )

    model = OpenaiChatModel("test", api_key="test")
    model._client = openai.OpenAI(
        api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handle))
    )
    result = model.complete([UserMessage("Hi")])
    assert result.content == "hi"
    assert result.usage == Usage(7, 4)


def test_nonstream_length_keeps_existing_text_behavior():
    payload = completion({"content": "truncated"})
    payload["choices"][0]["finish_reason"] = "length"
    model = model_for(payload, [])
    assert model.complete([UserMessage("Hi")]).content == "truncated"


def test_nonstream_missing_usage():
    payload = completion({"content": "hello"})
    del payload["usage"]
    model = model_for(payload, [])
    assert model.complete([UserMessage("Hi")]).usage is None


async def test_async_nonstream_structured_output():
    tool = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "return_result", "arguments": '{"value":5}'},
    }
    model = model_for(completion({"content": None, "tool_calls": [tool]}), [])
    result = await model.acomplete([UserMessage("Value")], output_types=[Result])
    assert result.content == Result(value=5)
    assert result.usage == Usage(7, 4)


def test_nonstream_api_error_propagates():
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"error": {"message": "Invalid key", "type": "authentication_error"}},
        )

    model = OpenaiChatModel("compatible-model", api_key="test", stream=False)
    model._client = openai.OpenAI(
        api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handle))
    )
    with pytest.raises(openai.AuthenticationError):
        model.complete([UserMessage("Hi")])


@pytest.mark.parametrize(
    "message",
    [{"content": None, "refusal": "Cannot help"}, {"content": ""}, {"content": None}],
)
@pytest.mark.parametrize("streaming", [False, True])
def test_empty_or_refusal_matches_streaming(message, streaming):
    payload = completion(message)

    def handle(request: httpx.Request) -> httpx.Response:
        if not streaming:
            return httpx.Response(200, json=payload)
        chunk = {
            "id": "x",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", **message},
                    "finish_reason": "stop",
                }
            ],
            "usage": payload["usage"],
        }
        return httpx.Response(
            200,
            text="data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n",
            headers={"content-type": "text/event-stream"},
        )

    model = OpenaiChatModel("test", api_key="test", stream=streaming)
    model._client = openai.OpenAI(
        api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handle))
    )
    with pytest.raises(StopIteration):
        model.complete([UserMessage("Hi")])


@pytest.mark.parametrize("streaming", [False, True])
def test_zero_choices_matches_streaming(streaming):
    payload = completion({"content": ""})
    payload["choices"] = []

    def handle(request: httpx.Request) -> httpx.Response:
        if not streaming:
            return httpx.Response(200, json=payload)
        chunk = {**payload, "object": "chat.completion.chunk"}
        return httpx.Response(
            200,
            text="data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n",
            headers={"content-type": "text/event-stream"},
        )

    model = OpenaiChatModel("test", api_key="test", stream=streaming)
    model._client = openai.OpenAI(
        api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handle))
    )
    with pytest.raises(StopIteration):
        model.complete([UserMessage("Hi")])


async def test_async_nonstream_mixed_text_and_tools():
    from magentic._streamed_response import AsyncStreamedResponse
    from magentic.streaming import AsyncStreamedStr

    tool = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "plus", "arguments": '{"a":2,"b":3}'},
    }
    model = model_for(completion({"content": "Adding", "tool_calls": [tool]}), [])
    result = await model.acomplete(
        [UserMessage("Add")], functions=[plus], output_types=[AsyncStreamedResponse]
    )
    import asyncio

    iterator = result.content.__aiter__()
    parts = [await asyncio.wait_for(anext(iterator), timeout=2) for _ in range(2)]
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(anext(iterator), timeout=2)
    assert isinstance(parts[0], AsyncStreamedStr)
    assert await parts[0].to_string() == "Adding"
    assert isinstance(parts[1], FunctionCall)
    assert parts[1]() == 5
    assert result.usage == Usage(7, 4)


def test_nonstream_mixed_text_and_tools_bounded():
    from itertools import islice

    from magentic._streamed_response import StreamedResponse

    tool = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "plus", "arguments": '{"a":2,"b":3}'},
    }
    model = model_for(completion({"content": "Adding", "tool_calls": [tool]}), [])
    result = model.complete(
        [UserMessage("Add")], functions=[plus], output_types=[StreamedResponse]
    )
    parts = list(islice(result.content, 3))
    assert len(parts) == 2
    assert str(parts[0]) == "Adding"
    assert isinstance(parts[1], FunctionCall)
    assert parts[1]() == 5
    assert result.usage == Usage(7, 4)


def test_nonstream_invalid_tool_arguments():
    from magentic.chat_model.base import ToolSchemaParseError

    tool = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "plus", "arguments": '{"a":"bad","b":3}'},
    }
    model = model_for(completion({"content": None, "tool_calls": [tool]}), [])
    with pytest.raises(ToolSchemaParseError):
        model.complete(
            [UserMessage("Add")], functions=[plus], output_types=[FunctionCall[int]]
        )
