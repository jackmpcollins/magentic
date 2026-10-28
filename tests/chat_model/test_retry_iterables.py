import json
from collections.abc import Callable, Iterable
from typing import Any, cast

import httpx
import openai
import pytest
from pydantic import BaseModel, ValidationError

from magentic.chat_model.base import ChatModel, OutputT, ToolSchemaParseError
from magentic.chat_model.message import AssistantMessage, Message, UserMessage
from magentic.chat_model.openai_chat_model import OpenaiChatModel
from magentic.chat_model.retry_chat_model import RetryChatModel


class Result(BaseModel):
    value: int


def tool() -> int:
    return 1


class ConsumingChatModel(ChatModel):
    def __init__(self) -> None:
        self.calls: list[tuple[Any, Any]] = []

    def complete(
        self,
        messages: Iterable[Message[Any]],
        functions: Iterable[Callable[..., Any]] | None = None,
        output_types: Iterable[type[OutputT]] | None = None,
        *,
        stop: list[str] | None = None,
    ) -> AssistantMessage[OutputT]:
        self.calls.append(
            (
                list(functions) if functions is not None else None,
                list(output_types) if output_types is not None else None,
            )
        )
        if len(self.calls) <= 2:
            try:
                Result.model_validate({"value": "bad"})
            except ValidationError as error:
                raise ToolSchemaParseError(
                    AssistantMessage("bad"), "call_id", error
                ) from error
        return AssistantMessage(cast(OutputT, Result(value=1)))

    async def acomplete(
        self,
        messages: Iterable[Message[Any]],
        functions: Iterable[Callable[..., Any]] | None = None,
        output_types: Iterable[type[OutputT]] | None = None,
        *,
        stop: list[str] | None = None,
    ) -> AssistantMessage[OutputT]:
        return self.complete(messages, functions, output_types, stop=stop)


@pytest.mark.parametrize("iterator_functions", [True, False])
@pytest.mark.parametrize("iterator_outputs", [True, False])
def test_retry_preserves_iterable_contracts(iterator_functions, iterator_outputs):
    model = ConsumingChatModel()
    retry = RetryChatModel(model, max_retries=2)
    retry.complete(
        iter([UserMessage("Return a result")]),
        functions=iter([tool]) if iterator_functions else [tool],
        output_types=iter([Result]) if iterator_outputs else [Result],
    )
    assert model.calls == [([tool], [Result])] * 3


@pytest.mark.parametrize("iterator_functions", [True, False])
@pytest.mark.parametrize("iterator_outputs", [True, False])
async def test_async_retry_preserves_iterable_contracts(
    iterator_functions, iterator_outputs
):
    model = ConsumingChatModel()
    retry = RetryChatModel(model, max_retries=2)
    await retry.acomplete(
        iter([UserMessage("Return a result")]),
        functions=iter([tool]) if iterator_functions else [tool],
        output_types=iter([Result]) if iterator_outputs else [Result],
    )
    assert model.calls == [([tool], [Result])] * 3


def test_retry_preserves_none_defaults():
    model = ConsumingChatModel()
    RetryChatModel(model, max_retries=2).complete([UserMessage("Return a result")])
    assert model.calls == [(None, None)] * 3


async def test_async_retry_preserves_none_defaults():
    model = ConsumingChatModel()
    await RetryChatModel(model, max_retries=2).acomplete(
        [UserMessage("Return a result")]
    )
    assert model.calls == [(None, None)] * 3


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_retry_preserves_tools_in_openai_requests(asynchronous):
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        value = "bad" if len(requests) == 1 else 1
        chunk = {
            "id": "completion",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_id",
                                "type": "function",
                                "function": {
                                    "name": "return_result",
                                    "arguments": json.dumps({"value": value}),
                                },
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
        )

    model = OpenaiChatModel("test", api_key="test")
    transport = httpx.MockTransport(respond)
    retry = RetryChatModel(model, max_retries=1)
    if asynchronous:
        async with httpx.AsyncClient(transport=transport) as async_client:
            model._async_client = openai.AsyncOpenAI(
                api_key="test", http_client=async_client
            )
            result = await retry.acomplete(
                [UserMessage("Return a result")],
                functions=iter([tool]),
                output_types=[Result],
            )
    else:
        with httpx.Client(transport=transport) as client:
            model._client = openai.OpenAI(api_key="test", http_client=client)
            result = retry.complete(
                [UserMessage("Return a result")],
                functions=iter([tool]),
                output_types=[Result],
            )
    assert result.content == Result(value=1)
    assert len(requests) == 2
    assert requests[0]["tools"] == requests[1]["tools"]
