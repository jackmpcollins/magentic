from collections.abc import AsyncIterator, Callable, Iterable, Iterator
from enum import Enum
from functools import singledispatch
from itertools import chain, groupby, takewhile
from typing import Any, Generic, Literal, TypeVar, cast, overload
from uuid import uuid4

import openai
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
from pydantic import ValidationError

from magentic.chat_model.base import ChatModel, StructuredOutputError
from magentic.chat_model.function_schema import (
    AsyncFunctionSchema,
    BaseFunctionSchema,
    FunctionCallFunctionSchema,
    FunctionSchema,
    async_function_schema_for_type,
    function_schema_for_type,
)
from magentic.chat_model.message import (
    AssistantMessage,
    FunctionResultMessage,
    Message,
    SystemMessage,
    UserMessage,
)
from magentic.function_call import (
    AsyncParallelFunctionCall,
    FunctionCall,
    ParallelFunctionCall,
)
from magentic.streaming import (
    AsyncStreamedStr,
    StreamedStr,
    achain,
    agroupby,
    async_iter,
    atakewhile,
)
from magentic.typing import is_any_origin_subclass, is_origin_subclass


class OpenaiMessageRole(Enum):
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"
    USER = "user"


@singledispatch
def message_to_openai_message(message: Message[Any]) -> ChatCompletionMessageParam:
    """Convert a Message to an OpenAI message."""
    # TODO: Add instructions for registering new Message type to this error message
    raise NotImplementedError(type(message))


@message_to_openai_message.register
def _(message: SystemMessage) -> ChatCompletionMessageParam:
    return {"role": OpenaiMessageRole.SYSTEM.value, "content": message.content}


@message_to_openai_message.register
def _(message: UserMessage) -> ChatCompletionMessageParam:
    return {"role": OpenaiMessageRole.USER.value, "content": message.content}


@message_to_openai_message.register(AssistantMessage)
def _(message: AssistantMessage[Any]) -> ChatCompletionMessageParam:
    if isinstance(message.content, str):
        return {"role": OpenaiMessageRole.ASSISTANT.value, "content": message.content}

    function_schema: FunctionSchema[Any]

    if isinstance(message.content, FunctionCall):
        function_schema = FunctionCallFunctionSchema(message.content.function)
        return {
            "role": OpenaiMessageRole.ASSISTANT.value,
            "content": None,
            "tool_calls": [
                {
                    "id": message.content._unique_id,
                    "type": "function",
                    "function": {
                        "name": function_schema.name,
                        "arguments": function_schema.serialize_args(message.content),
                    },
                }
            ],
        }

    if isinstance(message.content, ParallelFunctionCall):
        return {
            "role": OpenaiMessageRole.ASSISTANT.value,
            "content": None,
            "tool_calls": [
                {
                    "id": function_call._unique_id,
                    "type": "function",
                    "function": {
                        "name": FunctionCallFunctionSchema(function_call.function).name,
                        "arguments": FunctionCallFunctionSchema(
                            function_call.function
                        ).serialize_args(function_call),
                    },
                }
                for function_call in message.content
            ],
        }

    function_schema = function_schema_for_type(type(message.content))
    return {
        "role": OpenaiMessageRole.ASSISTANT.value,
        "content": None,
        "tool_calls": [
            {
                # Can be random because no result will be inserted back into the chat
                "id": str(uuid4()),
                "type": "function",
                "function": {
                    "name": function_schema.name,
                    "arguments": function_schema.serialize_args(message.content),
                },
            }
        ],
    }


@message_to_openai_message.register(FunctionResultMessage)
def _(message: FunctionResultMessage[Any]) -> ChatCompletionMessageParam:
    function_schema = function_schema_for_type(type(message.content))
    return {
        "role": OpenaiMessageRole.TOOL.value,
        "tool_call_id": message.function_call._unique_id,
        "content": function_schema.serialize_args(message.content),
    }


def _add_missing_tool_calls_responses(
    messages: list[ChatCompletionMessageParam],
) -> list[ChatCompletionMessageParam]:
    """Add null responses for tool calls without a response.

    This is required by OpenAI's API.
    "An assistant message with 'tool_calls' must be followed by tool messages responding to each 'tool_call_id'."
    """
    new_messages: list[ChatCompletionMessageParam] = []
    current_tool_call_responses: set[str] = set()
    for message in reversed(messages):
        if tool_call_id := message.get("tool_call_id"):
            current_tool_call_responses.add(tool_call_id)  # type: ignore[arg-type]
        elif tool_calls := message.get("tool_calls"):
            for tool_call in tool_calls:  # type: ignore[attr-defined]
                if tool_call["id"] not in current_tool_call_responses:
                    new_messages.append(
                        {
                            "role": OpenaiMessageRole.TOOL.value,
                            "tool_call_id": tool_call["id"],
                            "content": "null",
                        }
                    )
                    current_tool_call_responses.add(tool_call["id"])
            current_tool_call_responses = set()

        new_messages.append(message)

    return list(reversed(new_messages))


T = TypeVar("T")
BaseFunctionSchemaT = TypeVar("BaseFunctionSchemaT", bound=BaseFunctionSchema[Any])


class BaseFunctionToolSchema(Generic[BaseFunctionSchemaT]):
    def __init__(self, function_schema: BaseFunctionSchemaT):
        self._function_schema = function_schema

    def as_tool_choice(self) -> ChatCompletionToolChoiceOptionParam:
        return {"type": "function", "function": {"name": self._function_schema.name}}

    def to_dict(self) -> ChatCompletionToolParam:
        return {"type": "function", "function": self._function_schema.dict()}

    def matches(self, tool_call: ChoiceDeltaToolCall) -> bool:
        return bool(
            tool_call.type == "function"
            and tool_call.function
            and self._function_schema.name == tool_call.function.name
        )


class FunctionToolSchema(BaseFunctionToolSchema[FunctionSchema[T]]):
    def parse_tool_call(self, chunks: Iterator[ChoiceDeltaToolCall]) -> T:
        return self._function_schema.parse_args(
            chunk.function.arguments
            for chunk in chunks
            if chunk.function and chunk.function.arguments is not None
        )


class AsyncFunctionToolSchema(BaseFunctionToolSchema[AsyncFunctionSchema[T]]):
    async def aparse_tool_call(self, chunks: AsyncIterator[ChoiceDeltaToolCall]) -> T:
        return await self._function_schema.aparse_args(
            chunk.function.arguments
            async for chunk in chunks
            if chunk.function and chunk.function.arguments is not None
        )


def _iter_streamed_tool_calls(
    response: Iterator[ChatCompletionChunk],
) -> Iterator[Iterator[ChoiceDeltaToolCall]]:
    response = takewhile(lambda chunk: chunk.choices[0].delta.tool_calls, response)
    for _, tool_call_chunks in groupby(
        response,
        lambda chunk: chunk.choices[0].delta.tool_calls
        and chunk.choices[0].delta.tool_calls[0].index,
    ):
        yield (
            chunk.choices[0].delta.tool_calls[0]
            for chunk in tool_call_chunks
            if chunk.choices[0].delta.tool_calls
        )


async def _aiter_streamed_tool_calls(
    response: AsyncIterator[ChatCompletionChunk],
) -> AsyncIterator[AsyncIterator[ChoiceDeltaToolCall]]:
    response = atakewhile(lambda chunk: chunk.choices[0].delta.tool_calls, response)
    async for _, tool_call_chunks in agroupby(
        response,
        lambda chunk: chunk.choices[0].delta.tool_calls
        and chunk.choices[0].delta.tool_calls[0].index,
    ):
        yield (
            chunk.choices[0].delta.tool_calls[0]
            async for chunk in tool_call_chunks
            if chunk.choices[0].delta.tool_calls
        )


def openai_chatcompletion_create(
    api_key: str | None,
    api_type: Literal["openai", "azure"],
    base_url: str | None,
    model: str,
    messages: list[ChatCompletionMessageParam],
    max_tokens: int | None = None,
    seed: int | None = None,
    stop: list[str] | None = None,
    temperature: float | None = None,
    tools: list[ChatCompletionToolParam] | None = None,
    tool_choice: ChatCompletionToolChoiceOptionParam | None = None,
) -> Iterator[ChatCompletionChunk]:
    client_kwargs: dict[str, Any] = {
        "api_key": api_key,
    }
    if api_type == "openai" and base_url:
        client_kwargs["base_url"] = base_url

    client = (
        openai.AzureOpenAI(**client_kwargs)
        if api_type == "azure"
        else openai.OpenAI(**client_kwargs)
    )

    # `openai.OpenAI().chat.completions.create` doesn't accept `None` for some args
    # so only pass function args if there are functions
    kwargs: dict[str, Any] = {}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if stop is not None:
        kwargs["stop"] = stop
    if tools:
        kwargs["tools"] = tools
    if tool_choice:
        kwargs["tool_choice"] = tool_choice

    response: Iterator[ChatCompletionChunk] = client.chat.completions.create(
        model=model,
        messages=messages,
        seed=seed,
        stream=True,
        temperature=temperature,
        **kwargs,
    )
    return response


async def openai_chatcompletion_acreate(
    api_key: str | None,
    api_type: Literal["openai", "azure"],
    base_url: str | None,
    model: str,
    messages: list[ChatCompletionMessageParam],
    max_tokens: int | None = None,
    seed: int | None = None,
    stop: list[str] | None = None,
    temperature: float | None = None,
    tools: list[ChatCompletionToolParam] | None = None,
    tool_choice: ChatCompletionToolChoiceOptionParam | None = None,
) -> AsyncIterator[ChatCompletionChunk]:
    client_kwargs: dict[str, Any] = {
        "api_key": api_key,
    }
    if api_type == "openai" and base_url:
        client_kwargs["base_url"] = base_url

    client = (
        openai.AsyncAzureOpenAI(**client_kwargs)
        if api_type == "azure"
        else openai.AsyncOpenAI(**client_kwargs)
    )
    # `openai.AsyncOpenAI().chat.completions.create` doesn't accept `None` for some args
    # so only pass function args if there are functions
    kwargs: dict[str, Any] = {}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if stop is not None:
        kwargs["stop"] = stop
    if tools:
        kwargs["tools"] = tools
    if tool_choice:
        kwargs["tool_choice"] = tool_choice

    response: AsyncIterator[ChatCompletionChunk] = await client.chat.completions.create(
        model=model,
        messages=messages,
        seed=seed,
        temperature=temperature,
        stream=True,
        **kwargs,
    )
    return response


# TODO: Generalize this to BaseToolSchema when that is created
BeseToolSchemaT = TypeVar("BeseToolSchemaT", bound=BaseFunctionToolSchema[Any])
R = TypeVar("R")


class OpenaiChatModel(ChatModel):
    """An LLM chat model that uses the `openai` python package."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        api_type: Literal["openai", "azure"] = "openai",
        base_url: str | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        temperature: float | None = None,
    ):
        self._model = model
        self._api_key = api_key
        self._api_type = api_type
        self._base_url = base_url
        self._max_tokens = max_tokens
        self._seed = seed
        self._temperature = temperature

    @property
    def model(self) -> str:
        return self._model

    @property
    def api_key(self) -> str | None:
        return self._api_key

    @property
    def api_type(self) -> Literal["openai", "azure"]:
        return self._api_type

    @property
    def base_url(self) -> str | None:
        return self._base_url

    @property
    def max_tokens(self) -> int | None:
        return self._max_tokens

    @property
    def seed(self) -> int | None:
        return self._seed

    @property
    def temperature(self) -> float | None:
        return self._temperature

    @staticmethod
    def _select_tool_schema(
        tool_call: ChoiceDeltaToolCall, tools_schemas: list[BeseToolSchemaT]
    ) -> BeseToolSchemaT:
        """Select the tool schema based on the response chunk."""
        for tool_schema in tools_schemas:
            if tool_schema.matches(tool_call):
                return tool_schema

        msg = f"Unknown tool call: {tool_call.model_dump_json()}"
        raise ValueError(msg)

    @overload
    def complete(
        self,
        messages: Iterable[Message[Any]],
        functions: Any = ...,
        output_types: None = ...,
        *,
        stop: list[str] | None = ...,
    ) -> AssistantMessage[str]: ...

    @overload
    def complete(
        self,
        messages: Iterable[Message[Any]],
        functions: Any = ...,
        output_types: Iterable[type[R]] = ...,
        *,
        stop: list[str] | None = ...,
    ) -> AssistantMessage[R]: ...

    def complete(
        self,
        messages: Iterable[Message[Any]],
        functions: Iterable[Callable[..., Any]] | None = None,
        output_types: Iterable[type[R]] | None = None,
        *,
        stop: list[str] | None = None,
    ) -> AssistantMessage[str] | AssistantMessage[R]:
        """Request an LLM message."""
        if output_types is None:
            output_types = [] if functions else cast(list[type[R]], [str])

        # TODO: Check that Function calls types match functions
        function_schemas = [FunctionCallFunctionSchema(f) for f in functions or []] + [
            function_schema_for_type(type_)
            for type_ in output_types
            if not is_origin_subclass(
                type_, (str, StreamedStr, FunctionCall, ParallelFunctionCall)
            )
        ]
        tool_schemas = [FunctionToolSchema(schema) for schema in function_schemas]

        str_in_output_types = is_any_origin_subclass(output_types, str)
        streamed_str_in_output_types = is_any_origin_subclass(output_types, StreamedStr)
        allow_string_output = str_in_output_types or streamed_str_in_output_types

        response = openai_chatcompletion_create(
            api_key=self.api_key,
            api_type=self.api_type,
            base_url=self.base_url,
            model=self.model,
            messages=_add_missing_tool_calls_responses(
                [message_to_openai_message(m) for m in messages]
            ),
            max_tokens=self.max_tokens,
            seed=self.seed,
            stop=stop,
            temperature=self.temperature,
            tools=[schema.to_dict() for schema in tool_schemas],
            tool_choice=(
                tool_schemas[0].as_tool_choice()
                if len(tool_schemas) == 1 and not allow_string_output
                else None
            ),
        )

        first_chunk = next(response)
        # Azure OpenAI sends a chunk with empty choices first
        if len(first_chunk.choices) == 0:
            first_chunk = next(response)
        if (
            first_chunk.choices[0].delta.content is None
            and first_chunk.choices[0].delta.tool_calls is None
        ):
            first_chunk = next(response)
        response = chain([first_chunk], response)

        if first_chunk.choices[0].delta.content is not None:
            if not allow_string_output:
                msg = (
                    "String was returned by model but not expected. You may need to update"
                    " your prompt to encourage the model to return a specific type."
                )
                raise StructuredOutputError(msg)
            streamed_str = StreamedStr(
                chunk.choices[0].delta.content
                for chunk in response
                if chunk.choices[0].delta.content is not None
            )
            if streamed_str_in_output_types:
                return AssistantMessage(streamed_str)  # type: ignore[return-value]
            return AssistantMessage(str(streamed_str))

        if first_chunk.choices[0].delta.tool_calls is not None:
            if is_any_origin_subclass(output_types, ParallelFunctionCall):
                content = ParallelFunctionCall(
                    self._select_tool_schema(
                        next(tool_call_chunks), tool_schemas
                    ).parse_tool_call(tool_call_chunks)
                    for tool_call_chunks in _iter_streamed_tool_calls(response)
                )
                return AssistantMessage(content)  # type: ignore[return-value]

            tool_schema = self._select_tool_schema(
                first_chunk.choices[0].delta.tool_calls[0], tool_schemas
            )
            try:
                # Take only the first tool_call, silently ignore extra chunks
                # TODO: Create generator here that raises error or warns if multiple tool_calls
                content = tool_schema.parse_tool_call(
                    next(_iter_streamed_tool_calls(response))
                )
                return AssistantMessage(content)  # type: ignore[return-value]
            except ValidationError as e:
                msg = (
                    "Failed to parse model output. You may need to update your prompt"
                    " to encourage the model to return a specific type."
                )
                raise StructuredOutputError(msg) from e

        msg = f"Could not determine response type for first chunk: {first_chunk.model_dump_json()}"
        raise ValueError(msg)

    @overload
    async def acomplete(
        self,
        messages: Iterable[Message[Any]],
        functions: Any = ...,
        output_types: None = ...,
        *,
        stop: list[str] | None = ...,
    ) -> AssistantMessage[str]: ...

    @overload
    async def acomplete(
        self,
        messages: Iterable[Message[Any]],
        functions: Any = ...,
        output_types: Iterable[type[R]] = ...,
        *,
        stop: list[str] | None = ...,
    ) -> AssistantMessage[R]: ...

    async def acomplete(
        self,
        messages: Iterable[Message[Any]],
        functions: Iterable[Callable[..., Any]] | None = None,
        output_types: Iterable[type[R]] | None = None,
        *,
        stop: list[str] | None = None,
    ) -> AssistantMessage[R] | AssistantMessage[str]:
        """Async version of `complete`."""
        if output_types is None:
            output_types = [] if functions else cast(list[type[R]], [str])

        function_schemas = [FunctionCallFunctionSchema(f) for f in functions or []] + [
            async_function_schema_for_type(type_)
            for type_ in output_types
            if not is_origin_subclass(type_, (str, AsyncStreamedStr, FunctionCall))
        ]
        tool_schemas = [AsyncFunctionToolSchema(schema) for schema in function_schemas]

        str_in_output_types = is_any_origin_subclass(output_types, str)
        async_streamed_str_in_output_types = is_any_origin_subclass(
            output_types, AsyncStreamedStr
        )
        allow_string_output = str_in_output_types or async_streamed_str_in_output_types

        response = await openai_chatcompletion_acreate(
            api_key=self.api_key,
            api_type=self.api_type,
            base_url=self.base_url,
            model=self.model,
            messages=_add_missing_tool_calls_responses(
                [message_to_openai_message(m) for m in messages]
            ),
            max_tokens=self.max_tokens,
            seed=self.seed,
            stop=stop,
            temperature=self.temperature,
            tools=[schema.to_dict() for schema in tool_schemas],
            tool_choice=(
                tool_schemas[0].as_tool_choice()
                if len(tool_schemas) == 1 and not allow_string_output
                else None
            ),
        )

        first_chunk = await anext(response)
        # Azure OpenAI sends a chunk with empty choices first
        if len(first_chunk.choices) == 0:
            first_chunk = await anext(response)
        if (
            first_chunk.choices[0].delta.content is None
            and first_chunk.choices[0].delta.tool_calls is None
        ):
            first_chunk = await anext(response)
        response = achain(async_iter([first_chunk]), response)

        if first_chunk.choices[0].delta.content is not None:
            if not allow_string_output:
                msg = (
                    "String was returned by model but not expected. You may need to update"
                    " your prompt to encourage the model to return a specific type."
                )
                raise StructuredOutputError(msg)
            async_streamed_str = AsyncStreamedStr(
                chunk.choices[0].delta.content
                async for chunk in response
                if chunk.choices[0].delta.content is not None
            )
            if async_streamed_str_in_output_types:
                return AssistantMessage(async_streamed_str)  # type: ignore[return-value]
            return AssistantMessage(await async_streamed_str.to_string())

        if first_chunk.choices[0].delta.tool_calls is not None:
            if is_any_origin_subclass(output_types, AsyncParallelFunctionCall):
                content = AsyncParallelFunctionCall(
                    await self._select_tool_schema(
                        await anext(tool_call_chunks), tool_schemas
                    ).aparse_tool_call(tool_call_chunks)
                    async for tool_call_chunks in _aiter_streamed_tool_calls(response)
                )
                return AssistantMessage(content)  # type: ignore[return-value]

            tool_schema = self._select_tool_schema(
                first_chunk.choices[0].delta.tool_calls[0], tool_schemas
            )
            try:
                # Take only the first tool_call, silently ignore extra chunks
                content = await tool_schema.aparse_tool_call(
                    await anext(_aiter_streamed_tool_calls(response))
                )
                return AssistantMessage(content)  # type: ignore[return-value]
            except ValidationError as e:
                msg = (
                    "Failed to parse model output. You may need to update your prompt"
                    " to encourage the model to return a specific type."
                )
                raise StructuredOutputError(msg) from e

        msg = f"Could not determine response type for first chunk: {first_chunk.model_dump_json()}"
        raise ValueError(msg)
