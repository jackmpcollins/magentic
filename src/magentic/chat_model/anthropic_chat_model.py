import base64
import json
from collections.abc import AsyncIterator, Callable, Iterable, Iterator
from enum import Enum
from functools import singledispatch
from itertools import chain, groupby
from typing import Any, AsyncIterable, Generic, Sequence, TypeVar, cast, overload

import filetype
from pydantic import ValidationError

from magentic.chat_model.base import (
    ChatModel,
    ToolSchemaParseError,
    avalidate_str_content,
    validate_str_content,
)
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
    Message,
    SystemMessage,
    ToolResultMessage,
    Usage,
    UserMessage,
    _RawMessage,
)
from magentic.function_call import (
    AsyncParallelFunctionCall,
    FunctionCall,
    ParallelFunctionCall,
    _create_unique_id,
)
from magentic.streaming import (
    AsyncStreamedStr,
    StreamedStr,
    aapply,
    achain,
    agroupby,
    apeek,
    apply,
    async_iter,
    peek,
)
from magentic.typing import is_any_origin_subclass, is_origin_subclass
from magentic.vision import UserImageMessage

try:
    import anthropic
    from anthropic.lib.streaming import MessageStreamEvent
    from anthropic.lib.streaming._messages import accumulate_event
    from anthropic.types import (
        ContentBlockDeltaEvent,
        ContentBlockStartEvent,
        MessageParam,
        ToolParam,
        ToolUseBlock,
    )
    from anthropic.types.message_create_params import ToolChoice
except ImportError as error:
    msg = "To use AnthropicChatModel you must install the `anthropic` package using `pip install 'magentic[anthropic]'`."
    raise ImportError(msg) from error


class AnthropicMessageRole(Enum):
    ASSISTANT = "assistant"
    USER = "user"


@singledispatch
def message_to_anthropic_message(message: Message[Any]) -> MessageParam:
    """Convert a Message to an OpenAI message."""
    # TODO: Add instructions for registering new Message type to this error message
    raise NotImplementedError(type(message))


@message_to_anthropic_message.register(_RawMessage)
def _(message: _RawMessage[Any]) -> MessageParam:
    # TODO: Validate the message content
    return message.content  # type: ignore[no-any-return]


@message_to_anthropic_message.register
def _(message: UserMessage) -> MessageParam:
    return {"role": AnthropicMessageRole.USER.value, "content": message.content}


@message_to_anthropic_message.register(UserImageMessage)
def _(message: UserImageMessage[Any]) -> MessageParam:
    if isinstance(message.content, bytes):
        mime_type = filetype.guess_mime(message.content)
        base64_image = base64.b64encode(message.content).decode("utf-8")
    else:
        msg = f"Invalid content type: {type(message.content)}"
        raise TypeError(msg)

    return {
        "role": AnthropicMessageRole.USER.value,
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime_type,
                    "data": base64_image,
                },
            }
        ],
    }


@message_to_anthropic_message.register(AssistantMessage)
def _(message: AssistantMessage[Any]) -> MessageParam:
    if isinstance(message.content, str):
        return {
            "role": AnthropicMessageRole.ASSISTANT.value,
            "content": message.content,
        }

    function_schema: FunctionSchema[Any]

    if isinstance(message.content, FunctionCall):
        function_schema = FunctionCallFunctionSchema(message.content.function)
        return {
            "role": AnthropicMessageRole.ASSISTANT.value,
            "content": [
                {
                    "type": "tool_use",
                    "id": message.content._unique_id,
                    "name": function_schema.name,
                    "input": json.loads(
                        function_schema.serialize_args(message.content)
                    ),
                }
            ],
        }

    if isinstance(message.content, ParallelFunctionCall):
        return {
            "role": AnthropicMessageRole.ASSISTANT.value,
            "content": [
                {
                    "type": "tool_use",
                    "id": function_call._unique_id,
                    "name": FunctionCallFunctionSchema(function_call.function).name,
                    "input": json.loads(
                        FunctionCallFunctionSchema(
                            function_call.function
                        ).serialize_args(function_call)
                    ),
                }
                for function_call in message.content
            ],
        }

    function_schema = function_schema_for_type(type(message.content))
    return {
        "role": AnthropicMessageRole.ASSISTANT.value,
        "content": [
            {
                "type": "tool_use",
                # Can be random because no result will be inserted back into the chat
                "id": _create_unique_id(),
                "name": function_schema.name,
                "input": json.loads(function_schema.serialize_args(message.content)),
            }
        ],
    }


@message_to_anthropic_message.register(ToolResultMessage)
def _(message: ToolResultMessage[Any]) -> MessageParam:
    if isinstance(message.content, str):
        content = message.content
    else:
        function_schema = function_schema_for_type(type(message.content))
        content = json.loads(function_schema.serialize_args(message.content))
    return {
        "role": AnthropicMessageRole.USER.value,
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": message.tool_call_id,
                "content": content,
            }
        ],
    }


# TODO: Move this to the magentic level by allowing `UserMessage` have a list of content
def _combine_messages(messages: Iterable[MessageParam]) -> list[MessageParam]:
    """Combine messages with the same role, to get alternating roles.

    Alternating roles is a requirement of the Anthropic API.
    """
    combined_messages: list[MessageParam] = []
    for message_group in groupby(messages, lambda x: x["role"]):
        role, messages = message_group
        content = []
        for message in messages:
            if isinstance(message["content"], list):
                content.extend(message["content"])
            elif isinstance(message["content"], str):
                content.append({"type": "text", "text": message["content"]})
            else:
                content.append(message["content"])
        combined_messages.append({"role": role, "content": content})
    return combined_messages


T = TypeVar("T")
BaseFunctionSchemaT = TypeVar("BaseFunctionSchemaT", bound=BaseFunctionSchema[Any])


class BaseFunctionToolSchema(Generic[BaseFunctionSchemaT]):
    def __init__(self, function_schema: BaseFunctionSchemaT):
        self._function_schema = function_schema

    def to_dict(self) -> ToolParam:
        return {
            "name": self._function_schema.name,
            "description": self._function_schema.description or "",
            "input_schema": self._function_schema.parameters,
        }

    def as_tool_choice(self) -> ToolChoice:
        return {"type": "tool", "name": self._function_schema.name}


# TODO: Generalize this to BaseToolSchema when that is created
BeseToolSchemaT = TypeVar("BeseToolSchemaT", bound=BaseFunctionToolSchema[Any])


def select_tool_schema(
    tool_call: ToolUseBlock,
    tool_schemas: Iterable[BeseToolSchemaT],
) -> BeseToolSchemaT:
    """Select the tool schema based on the response chunk."""
    for tool_schema in tool_schemas:
        if tool_schema._function_schema.name == tool_call.name:
            return tool_schema

    msg = f"Unknown tool call: {tool_call.model_dump_json()}"
    raise ValueError(msg)


class FunctionToolSchema(BaseFunctionToolSchema[FunctionSchema[T]]):
    def parse_tool_call(self, chunks: Iterable[MessageStreamEvent]) -> T:
        return self._function_schema.parse_args(
            chunk.delta.partial_json
            for chunk in chunks
            if chunk.type == "content_block_delta"
            if chunk.delta.type == "input_json_delta"
        )


class AsyncFunctionToolSchema(BaseFunctionToolSchema[AsyncFunctionSchema[T]]):
    async def aparse_tool_call(self, chunks: AsyncIterable[MessageStreamEvent]) -> T:
        return await self._function_schema.aparse_args(
            chunk.delta.partial_json
            async for chunk in chunks
            if chunk.type == "content_block_delta"
            if chunk.delta.type == "input_json_delta"
        )


def _iter_streamed_tool_calls(
    response: Iterable[MessageStreamEvent],
) -> Iterator[Iterator[ContentBlockStartEvent | ContentBlockDeltaEvent]]:
    all_tool_call_chunks = (
        cast(ContentBlockStartEvent | ContentBlockDeltaEvent, chunk)
        for chunk in response
        if chunk.type in ("content_block_start", "content_block_delta")
    )
    for _, tool_call_chunks in groupby(all_tool_call_chunks, lambda x: x.index):
        yield tool_call_chunks


async def _aiter_streamed_tool_calls(
    response: AsyncIterable[MessageStreamEvent],
) -> AsyncIterator[AsyncIterator[ContentBlockStartEvent | ContentBlockDeltaEvent]]:
    all_tool_call_chunks = (
        cast(ContentBlockStartEvent | ContentBlockDeltaEvent, chunk)
        async for chunk in response
        if chunk.type in ("content_block_start", "content_block_delta")
    )
    async for _, tool_call_chunks in agroupby(all_tool_call_chunks, lambda x: x.index):
        yield tool_call_chunks


def _join_streamed_response_to_message(
    response: list[MessageStreamEvent],
) -> _RawMessage[MessageParam]:
    snapshot = None
    for event in response:
        snapshot = accumulate_event(
            event=event,  # type: ignore[arg-type]
            current_snapshot=snapshot,
        )
    assert snapshot is not None  # noqa: S101
    snapshot_content = snapshot.model_dump()["content"]
    return _RawMessage({"role": snapshot.role, "content": snapshot_content})


def _parse_streamed_tool_calls(
    response: Iterable[MessageStreamEvent],
    tool_schemas: Iterable[FunctionToolSchema[T]],
) -> Iterator[T]:
    cached_response: list[MessageStreamEvent] = []
    response = apply(cached_response.append, response)
    try:
        for tool_call_chunks in _iter_streamed_tool_calls(response):
            first_chunk, tool_call_chunks = peek(tool_call_chunks)
            assert first_chunk.type == "content_block_start"  # noqa: S101
            assert first_chunk.content_block.type == "tool_use"  # noqa: S101
            tool_schema = select_tool_schema(first_chunk.content_block, tool_schemas)
            tool_call = tool_schema.parse_tool_call(tool_call_chunks)
            yield tool_call
    # TODO: Catch/raise unknown tool call error here
    except ValidationError as e:
        raw_message = _join_streamed_response_to_message(cached_response)
        raise ToolSchemaParseError(
            output_message=raw_message,
            tool_call_id=raw_message.content["content"][0]["id"],  # type: ignore[index,unused-ignore]
            validation_error=e,
        ) from e


async def _aparse_streamed_tool_calls(
    response: AsyncIterable[MessageStreamEvent],
    tool_schemas: Iterable[AsyncFunctionToolSchema[T]],
) -> AsyncIterator[T]:
    cached_response: list[MessageStreamEvent] = []
    response = aapply(cached_response.append, response)
    try:
        async for tool_call_chunks in _aiter_streamed_tool_calls(response):
            first_chunk, tool_call_chunks = await apeek(tool_call_chunks)
            assert first_chunk.type == "content_block_start"  # noqa: S101
            assert first_chunk.content_block.type == "tool_use"  # noqa: S101
            tool_schema = select_tool_schema(first_chunk.content_block, tool_schemas)
            tool_call = await tool_schema.aparse_tool_call(tool_call_chunks)
            yield tool_call
    # TODO: Catch/raise unknown tool call error here
    except ValidationError as e:
        raw_message = _join_streamed_response_to_message(cached_response)
        raise ToolSchemaParseError(
            output_message=raw_message,
            tool_call_id=raw_message.content["content"][0]["id"],  # type: ignore[index,unused-ignore]
            validation_error=e,
        ) from e


def _extract_system_message(
    messages: Iterable[Message[Any]],
) -> tuple[str | anthropic.NotGiven, list[Message[Any]]]:
    system_messages = [m for m in messages if isinstance(m, SystemMessage)]
    if len(system_messages) > 1:
        msg = "Only one system message is allowed per request."
        raise ValueError(msg)
    return (
        system_messages[0].content if system_messages else anthropic.NOT_GIVEN,
        [m for m in messages if not isinstance(m, SystemMessage)],
    )


def _create_usage_ref(
    response: Iterable[MessageStreamEvent],
) -> tuple[list[Usage], Iterator[MessageStreamEvent]]:
    """Returns a pointer to a Usage object that is created at the end of the response."""
    usage_ref: list[Usage] = []

    def generator(
        response: Iterable[MessageStreamEvent],
    ) -> Iterator[MessageStreamEvent]:
        message_start_usage = None
        output_tokens = None
        for chunk in response:
            if chunk.type == "message_start":
                message_start_usage = chunk.message.usage
            if chunk.type == "message_delta":
                output_tokens = chunk.usage.output_tokens
            yield chunk
        if message_start_usage and output_tokens:
            usage_ref.append(
                Usage(
                    input_tokens=message_start_usage.input_tokens,
                    output_tokens=message_start_usage.output_tokens + output_tokens,
                )
            )

    return usage_ref, generator(response)


def _create_usage_ref_async(
    response: AsyncIterable[MessageStreamEvent],
) -> tuple[list[Usage], AsyncIterator[MessageStreamEvent]]:
    """Async version of `_create_usage_ref`."""
    usage_ref: list[Usage] = []

    async def agenerator(
        response: AsyncIterable[MessageStreamEvent],
    ) -> AsyncIterator[MessageStreamEvent]:
        message_start_usage = None
        output_tokens = None
        async for chunk in response:
            if chunk.type == "message_start":
                message_start_usage = chunk.message.usage
            if chunk.type == "message_delta":
                output_tokens = chunk.usage.output_tokens
            yield chunk
        if message_start_usage and output_tokens:
            usage_ref.append(
                Usage(
                    input_tokens=message_start_usage.input_tokens,
                    output_tokens=message_start_usage.output_tokens + output_tokens,
                )
            )

    return usage_ref, agenerator(response)


R = TypeVar("R")


STR_OR_FUNCTIONCALL_TYPE = (
    str,
    StreamedStr,
    AsyncStreamedStr,
    FunctionCall,
    ParallelFunctionCall,
    AsyncParallelFunctionCall,
)


class AnthropicChatModel(ChatModel):
    """An LLM chat model that uses the `anthropic` python package."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 1024,
        temperature: float | None = None,
    ):
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._max_tokens = max_tokens
        self._temperature = temperature

        self._client = anthropic.Anthropic(api_key=api_key, base_url=base_url)
        self._async_client = anthropic.AsyncAnthropic(
            api_key=api_key, base_url=base_url
        )

    @property
    def model(self) -> str:
        return self._model

    @property
    def api_key(self) -> str | None:
        return self._api_key

    @property
    def base_url(self) -> str | None:
        return self._base_url

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @property
    def temperature(self) -> float | None:
        return self._temperature

    @staticmethod
    def _get_tool_choice(
        *,
        tool_schemas: Sequence[BaseFunctionToolSchema[Any]],
        allow_string_output: bool,
    ) -> ToolChoice | anthropic.NotGiven:
        """Create the tool choice argument."""
        if allow_string_output:
            return anthropic.NOT_GIVEN
        if len(tool_schemas) == 1:
            return tool_schemas[0].as_tool_choice()
        return {"type": "any"}

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
            if not is_origin_subclass(type_, STR_OR_FUNCTIONCALL_TYPE)
        ]
        tool_schemas = [FunctionToolSchema(schema) for schema in function_schemas]

        str_in_output_types = is_any_origin_subclass(output_types, str)
        streamed_str_in_output_types = is_any_origin_subclass(output_types, StreamedStr)
        allow_string_output = str_in_output_types or streamed_str_in_output_types

        system, messages = _extract_system_message(messages)

        def _response_generator() -> Iterator[MessageStreamEvent]:
            with self._client.messages.stream(
                model=self.model,
                messages=_combine_messages(
                    [message_to_anthropic_message(m) for m in messages]
                ),
                max_tokens=self.max_tokens,
                stop_sequences=stop or anthropic.NOT_GIVEN,
                system=system,
                temperature=(
                    self.temperature
                    if self.temperature is not None
                    else anthropic.NOT_GIVEN
                ),
                tools=(
                    [schema.to_dict() for schema in tool_schemas] or anthropic.NOT_GIVEN
                ),
                tool_choice=self._get_tool_choice(
                    tool_schemas=tool_schemas, allow_string_output=allow_string_output
                ),
            ) as stream:
                yield from stream

        response = _response_generator()
        usage_ref, response = _create_usage_ref(response)

        message_start_chunk = next(response)
        assert message_start_chunk.type == "message_start"  # noqa: S101
        first_chunk = next(response)
        assert first_chunk.type == "content_block_start"  # noqa: S101
        response = chain([message_start_chunk, first_chunk], response)

        if (
            first_chunk.type == "content_block_start"
            and first_chunk.content_block.type == "text"
        ):
            streamed_str = StreamedStr(
                chunk.delta.text
                for chunk in response
                if chunk.type == "content_block_delta"
                and chunk.delta.type == "text_delta"
            )
            str_content = validate_str_content(
                streamed_str,
                allow_string_output=allow_string_output,
                streamed=streamed_str_in_output_types,
            )
            return AssistantMessage._with_usage(str_content, usage_ref)  # type: ignore[return-value]

        if (
            first_chunk.type == "content_block_start"
            and first_chunk.content_block.type == "tool_use"
        ):
            tool_calls = _parse_streamed_tool_calls(response, tool_schemas)
            if is_any_origin_subclass(output_types, ParallelFunctionCall):
                content = ParallelFunctionCall(tool_calls)
                return AssistantMessage._with_usage(content, usage_ref)  # type: ignore[return-value]
            # Take only the first tool_call, silently ignore extra chunks
            # TODO: Create generator here that raises error or warns if multiple tool_calls
            content = next(tool_calls)
            return AssistantMessage._with_usage(content, usage_ref)  # type: ignore[return-value]

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
            if not is_origin_subclass(type_, STR_OR_FUNCTIONCALL_TYPE)
        ]
        tool_schemas = [AsyncFunctionToolSchema(schema) for schema in function_schemas]

        str_in_output_types = is_any_origin_subclass(output_types, str)
        async_streamed_str_in_output_types = is_any_origin_subclass(
            output_types, AsyncStreamedStr
        )
        allow_string_output = str_in_output_types or async_streamed_str_in_output_types

        system, messages = _extract_system_message(messages)

        async def _response_generator() -> AsyncIterator[MessageStreamEvent]:
            async with self._async_client.messages.stream(
                model=self.model,
                messages=_combine_messages(
                    [message_to_anthropic_message(m) for m in messages]
                ),
                max_tokens=self.max_tokens,
                stop_sequences=stop or anthropic.NOT_GIVEN,
                system=system,
                temperature=(
                    self.temperature
                    if self.temperature is not None
                    else anthropic.NOT_GIVEN
                ),
                tools=(
                    [schema.to_dict() for schema in tool_schemas] or anthropic.NOT_GIVEN
                ),
                tool_choice=self._get_tool_choice(
                    tool_schemas=tool_schemas, allow_string_output=allow_string_output
                ),
            ) as stream:
                async for chunk in stream:
                    yield chunk

        response = _response_generator()
        usage_ref, response = _create_usage_ref_async(response)

        message_start_chunk = await anext(response)
        assert message_start_chunk.type == "message_start"  # noqa: S101
        first_chunk = await anext(response)
        assert first_chunk.type == "content_block_start"  # noqa: S101
        response = achain(async_iter([message_start_chunk, first_chunk]), response)

        if (
            first_chunk.type == "content_block_start"
            and first_chunk.content_block.type == "text"
        ):
            async_streamed_str = AsyncStreamedStr(
                chunk.delta.text
                async for chunk in response
                if chunk.type == "content_block_delta"
                and chunk.delta.type == "text_delta"
            )
            str_content = await avalidate_str_content(
                async_streamed_str,
                allow_string_output=allow_string_output,
                streamed=async_streamed_str_in_output_types,
            )
            return AssistantMessage._with_usage(str_content, usage_ref)  # type: ignore[return-value]

        if (
            first_chunk.type == "content_block_start"
            and first_chunk.content_block.type == "tool_use"
        ):
            tool_calls = _aparse_streamed_tool_calls(response, tool_schemas)
            if is_any_origin_subclass(output_types, AsyncParallelFunctionCall):
                content = AsyncParallelFunctionCall(tool_calls)
                return AssistantMessage._with_usage(content, usage_ref)  # type: ignore[return-value]
            # Take only the first tool_call, silently ignore extra chunks
            # TODO: Create generator here that raises error or warns if multiple tool_calls
            content = await anext(tool_calls)
            return AssistantMessage._with_usage(content, usage_ref)  # type: ignore[return-value]

        msg = "Could not determine response type"
        raise ValueError(msg)
