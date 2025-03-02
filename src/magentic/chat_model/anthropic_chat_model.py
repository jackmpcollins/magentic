import json
from collections.abc import AsyncIterator, Callable, Iterable, Iterator, Sequence
from enum import Enum
from functools import singledispatch
from itertools import groupby
from typing import Any, Generic, cast

from typing_extensions import TypeVar

from magentic._parsing import contains_parallel_function_call_type, contains_string_type
from magentic._streamed_response import AsyncStreamedResponse, StreamedResponse
from magentic.chat_model.base import ChatModel, OutputT, aparse_stream, parse_stream
from magentic.chat_model.function_schema import (
    BaseFunctionSchema,
    FunctionCallFunctionSchema,
    function_schema_for_type,
    get_async_function_schemas,
    get_function_schemas,
)
from magentic.chat_model.message import (
    AssistantMessage,
    DocumentBytes,
    ImageBytes,
    Message,
    SystemMessage,
    ToolResultMessage,
    Usage,
    UserMessage,
    _RawMessage,
)
from magentic.chat_model.stream import (
    AsyncOutputStream,
    FunctionCallChunk,
    OutputStream,
    StreamParser,
    StreamState,
)
from magentic.function_call import (
    AsyncParallelFunctionCall,
    FunctionCall,
    ParallelFunctionCall,
    _create_unique_id,
)
from magentic.streaming import AsyncStreamedStr, StreamedStr
from magentic.vision import UserImageMessage

try:
    import anthropic
    from anthropic.lib.streaming import MessageStreamEvent
    from anthropic.lib.streaming._messages import accumulate_event
    from anthropic.types import (
        DocumentBlockParam,
        ImageBlockParam,
        MessageParam,
        TextBlockParam,
        ToolChoiceParam,
        ToolChoiceToolParam,
        ToolParam,
        ToolUseBlockParam,
    )
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


@singledispatch
async def async_message_to_anthropic_message(message: Message[Any]) -> MessageParam:
    """Async version of `message_to_anthropic_message`."""
    return message_to_anthropic_message(message)


@message_to_anthropic_message.register(_RawMessage)
def _(message: _RawMessage[Any]) -> MessageParam:
    # TODO: Validate the message content
    return message.content  # type: ignore[no-any-return]


@message_to_anthropic_message.register(UserMessage)
def _(message: UserMessage[Any]) -> MessageParam:
    if isinstance(message.content, str):
        return {"role": AnthropicMessageRole.USER.value, "content": message.content}
    if isinstance(message.content, Iterable):
        content: list[TextBlockParam | DocumentBlockParam | ImageBlockParam] = []
        for block in message.content:
            if isinstance(block, str):
                content.append({"type": "text", "text": block})
            elif isinstance(block, DocumentBytes):
                content.append(
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": block.mime_type,
                            "data": block.as_base64(),
                        },
                    }
                )
            elif isinstance(block, ImageBytes):
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": block.mime_type,
                            "data": block.as_base64(),
                        },
                    }
                )
            else:
                msg = f"Invalid content type for UserMessage: {type(block)}"
                raise TypeError(msg)
        return {"role": AnthropicMessageRole.USER.value, "content": content}
    msg = f"Invalid content type for UserMessage: {type(message.content)}"
    raise TypeError(msg)


@message_to_anthropic_message.register(UserImageMessage)
def _(message: UserImageMessage[Any]) -> MessageParam:
    if not isinstance(message.content, bytes):
        msg = f"Invalid content type: {type(message.content)}"
        raise TypeError(msg)

    image_bytes = ImageBytes(message.content)
    return {
        "role": AnthropicMessageRole.USER.value,
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image_bytes.mime_type,
                    "data": image_bytes.as_base64(),
                },
            }
        ],
    }


def _function_call_to_tool_call_block(
    function_call: FunctionCall[Any],
) -> ToolUseBlockParam:
    function_schema = FunctionCallFunctionSchema(function_call.function)
    return {
        "type": "tool_use",
        "id": function_call._unique_id,
        "name": function_schema.name,
        "input": json.loads(function_schema.serialize_args(function_call)),
    }


@message_to_anthropic_message.register(AssistantMessage)
def _(message: AssistantMessage[Any]) -> MessageParam:
    if isinstance(message.content, str):
        return {
            "role": AnthropicMessageRole.ASSISTANT.value,
            "content": message.content,
        }

    if isinstance(message.content, FunctionCall):
        return {
            "role": AnthropicMessageRole.ASSISTANT.value,
            "content": [_function_call_to_tool_call_block(message.content)],
        }

    if isinstance(message.content, ParallelFunctionCall):
        return {
            "role": AnthropicMessageRole.ASSISTANT.value,
            "content": [
                _function_call_to_tool_call_block(function_call)
                for function_call in message.content
            ],
        }

    if isinstance(message.content, StreamedResponse):
        content_blocks: list[TextBlockParam | ToolUseBlockParam] = []
        for item in message.content:
            if isinstance(item, StreamedStr):
                content_blocks.append({"type": "text", "text": item.to_string()})
            elif isinstance(item, FunctionCall):
                content_blocks.append(_function_call_to_tool_call_block(item))
        return {
            "role": AnthropicMessageRole.ASSISTANT.value,
            "content": content_blocks,
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


@async_message_to_anthropic_message.register(AssistantMessage)
async def _(message: AssistantMessage[Any]) -> MessageParam:
    if isinstance(message.content, AsyncParallelFunctionCall):
        return {
            "role": AnthropicMessageRole.ASSISTANT.value,
            "content": [
                _function_call_to_tool_call_block(function_call)
                async for function_call in message.content
            ],
        }

    if isinstance(message.content, AsyncStreamedResponse):
        content_blocks: list[TextBlockParam | ToolUseBlockParam] = []
        async for item in message.content:
            if isinstance(item, AsyncStreamedStr):
                content_blocks.append({"type": "text", "text": await item.to_string()})
            elif isinstance(item, FunctionCall):
                content_blocks.append(_function_call_to_tool_call_block(item))
        return {
            "role": AnthropicMessageRole.ASSISTANT.value,
            "content": content_blocks,
        }
    return message_to_anthropic_message(message)


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

    def as_tool_choice(self, *, disable_parallel_tool_use: bool) -> ToolChoiceToolParam:
        return {"type": "tool", "name": self._function_schema.name}


class AnthropicStreamParser(StreamParser[MessageStreamEvent]):
    def is_content(self, item: MessageStreamEvent) -> bool:
        return item.type == "content_block_delta"

    def get_content(self, item: MessageStreamEvent) -> str | None:
        if item.type == "text":
            return item.text
        return None

    def is_tool_call(self, item: MessageStreamEvent) -> bool:
        return (
            item.type == "content_block_start" and item.content_block.type == "tool_use"
        )

    def iter_tool_calls(self, item: MessageStreamEvent) -> Iterable[FunctionCallChunk]:
        if item.type == "content_block_start" and item.content_block.type == "tool_use":
            return [
                FunctionCallChunk(
                    id=item.content_block.id, name=item.content_block.name, args=None
                )
            ]
        if item.type == "input_json":
            return [FunctionCallChunk(id=None, name=None, args=item.partial_json)]
        return []


class AnthropicStreamState(StreamState[MessageStreamEvent]):
    def __init__(self) -> None:
        self._current_message_snapshot: anthropic.types.Message | None = (
            None  # TODO: type
        )
        self.usage_ref: list[Usage] = []

    def update(self, item: MessageStreamEvent) -> None:
        self._current_message_snapshot = accumulate_event(
            # Unrecognized event types are ignored
            event=item,  # type: ignore[arg-type]
            current_snapshot=self._current_message_snapshot,
        )
        if item.type == "message_stop":
            assert not self.usage_ref
            self.usage_ref.append(
                Usage(
                    input_tokens=item.message.usage.input_tokens,
                    output_tokens=item.message.usage.output_tokens,
                )
            )

    @property
    def current_message_snapshot(self) -> Message[Any]:
        assert self._current_message_snapshot is not None
        # TODO: Possible to return AssistantMessage here?
        return _RawMessage(self._current_message_snapshot.model_dump())


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


def _if_given(value: T | None) -> T | anthropic.NotGiven:
    return value if value is not None else anthropic.NOT_GIVEN


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
        output_types: Iterable[type],
    ) -> ToolChoiceParam | anthropic.NotGiven:
        """Create the tool choice argument."""
        if contains_string_type(output_types):
            return anthropic.NOT_GIVEN
        disable_parallel_tool_use = not contains_parallel_function_call_type(
            output_types
        )
        if len(tool_schemas) == 1:
            return tool_schemas[0].as_tool_choice(
                disable_parallel_tool_use=disable_parallel_tool_use
            )
        return {"type": "any", "disable_parallel_tool_use": disable_parallel_tool_use}

    def complete(
        self,
        messages: Iterable[Message[Any]],
        functions: Iterable[Callable[..., Any]] | None = None,
        output_types: Iterable[type[OutputT]] | None = None,
        *,
        stop: list[str] | None = None,
    ) -> AssistantMessage[OutputT]:
        """Request an LLM message."""
        if output_types is None:
            output_types = [] if functions else cast(list[type[OutputT]], [str])

        function_schemas = get_function_schemas(functions, output_types)
        tool_schemas = [BaseFunctionToolSchema(schema) for schema in function_schemas]

        system, messages = _extract_system_message(messages)

        response: Iterator[MessageStreamEvent] = self._client.messages.stream(
            model=self.model,
            messages=_combine_messages(
                [message_to_anthropic_message(m) for m in messages]
            ),
            max_tokens=self.max_tokens,
            stop_sequences=_if_given(stop),
            system=system,
            temperature=_if_given(self.temperature),
            tools=[schema.to_dict() for schema in tool_schemas] or anthropic.NOT_GIVEN,
            tool_choice=self._get_tool_choice(
                tool_schemas=tool_schemas, output_types=output_types
            ),
        ).__enter__()
        stream = OutputStream(
            response,
            function_schemas=function_schemas,
            parser=AnthropicStreamParser(),
            state=AnthropicStreamState(),
        )
        return AssistantMessage._with_usage(
            parse_stream(stream, output_types), usage_ref=stream.usage_ref
        )

    async def acomplete(
        self,
        messages: Iterable[Message[Any]],
        functions: Iterable[Callable[..., Any]] | None = None,
        output_types: Iterable[type[OutputT]] | None = None,
        *,
        stop: list[str] | None = None,
    ) -> AssistantMessage[OutputT]:
        """Async version of `complete`."""
        if output_types is None:
            output_types = [] if functions else cast(list[type[OutputT]], [str])

        function_schemas = get_async_function_schemas(functions, output_types)
        tool_schemas = [BaseFunctionToolSchema(schema) for schema in function_schemas]

        system, messages = _extract_system_message(messages)

        response: AsyncIterator[
            MessageStreamEvent
        ] = await self._async_client.messages.stream(
            model=self.model,
            messages=_combine_messages(
                [await async_message_to_anthropic_message(m) for m in messages]
            ),
            max_tokens=self.max_tokens,
            stop_sequences=_if_given(stop),
            system=system,
            temperature=_if_given(self.temperature),
            tools=[schema.to_dict() for schema in tool_schemas] or anthropic.NOT_GIVEN,
            tool_choice=self._get_tool_choice(
                tool_schemas=tool_schemas, output_types=output_types
            ),
        ).__aenter__()
        stream = AsyncOutputStream(
            response,
            function_schemas=function_schemas,
            parser=AnthropicStreamParser(),
            state=AnthropicStreamState(),
        )
        return AssistantMessage._with_usage(
            await aparse_stream(stream, output_types), usage_ref=stream.usage_ref
        )
