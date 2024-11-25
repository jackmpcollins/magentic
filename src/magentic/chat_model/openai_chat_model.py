import base64
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Sequence,
)
from enum import Enum
from functools import singledispatch
from typing import Any, Generic, Literal, TypeGuard, TypeVar, cast, overload

import filetype
import openai
from openai.lib.streaming.chat import (
    AsyncChatCompletionStreamManager,
    ChatCompletionStreamEvent,
    ChunkEvent,
    ContentDeltaEvent,
    ContentDoneEvent,
    FunctionToolCallArgumentsDeltaEvent,
    FunctionToolCallArgumentsDoneEvent,
)
from openai.lib.streaming.chat._completions import ChatCompletionStreamState
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionStreamOptionsParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
)

from magentic.chat_model.base import (
    ChatModel,
    aparse_stream,
    parse_stream,
)
from magentic.chat_model.function_schema import (
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
from magentic.chat_model.stream import (
    AsyncOutputStream,
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
from magentic.streaming import (
    AsyncStreamedStr,
    StreamedStr,
)
from magentic.typing import is_any_origin_subclass, is_origin_subclass
from magentic.vision import UserImageMessage


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


@message_to_openai_message.register(_RawMessage)
def _(message: _RawMessage[Any]) -> ChatCompletionMessageParam:
    # TODO: Validate the message content
    return message.content  # type: ignore[no-any-return]


@message_to_openai_message.register
def _(message: SystemMessage) -> ChatCompletionMessageParam:
    return {"role": OpenaiMessageRole.SYSTEM.value, "content": message.content}


@message_to_openai_message.register
def _(message: UserMessage) -> ChatCompletionMessageParam:
    return {"role": OpenaiMessageRole.USER.value, "content": message.content}


@message_to_openai_message.register(UserImageMessage)
def _(message: UserImageMessage[Any]) -> ChatCompletionMessageParam:
    if isinstance(message.content, bytes):
        mime_type = filetype.guess_mime(message.content)
        base64_image = base64.b64encode(message.content).decode("utf-8")
        url = f"data:{mime_type};base64,{base64_image}"
    elif isinstance(message.content, str):
        url = message.content
    else:
        msg = f"Invalid content type: {type(message.content)}"
        raise TypeError(msg)

    return {
        "role": OpenaiMessageRole.USER.value,
        "content": [{"type": "image_url", "image_url": {"url": url, "detail": "auto"}}],
    }


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
                "id": _create_unique_id(),
                "type": "function",
                "function": {
                    "name": function_schema.name,
                    "arguments": function_schema.serialize_args(message.content),
                },
            }
        ],
    }


@message_to_openai_message.register(ToolResultMessage)
def _(message: ToolResultMessage[Any]) -> ChatCompletionMessageParam:
    if isinstance(message.content, str):
        content = message.content
    else:
        function_schema = function_schema_for_type(type(message.content))
        content = function_schema.serialize_args(message.content)
    return {
        "role": OpenaiMessageRole.TOOL.value,
        "tool_call_id": message.tool_call_id,
        "content": content,
    }


# TODO: Use ToolResultMessage to solve this at magentic level
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


class OpenaiContentStreamParser(StreamParser[ChatCompletionStreamEvent, str]):
    """Filters and transforms OpenAI content events from a stream."""

    def is_member(
        self, item: ChatCompletionStreamEvent
    ) -> TypeGuard[ContentDeltaEvent]:
        return item.type == "content.delta"

    def is_end(self, item: ChatCompletionStreamEvent) -> TypeGuard[ContentDoneEvent]:
        return item.type == "content.done"

    def transform(self, item: ChatCompletionStreamEvent) -> str:
        assert self.is_member(item)  # noqa: S101
        return item.delta


class OpenaiToolStreamParser(StreamParser[ChatCompletionStreamEvent, str]):
    """Filters and transforms OpenAI tool events from a stream."""

    def is_member(
        self, item: ChatCompletionStreamEvent
    ) -> TypeGuard[FunctionToolCallArgumentsDeltaEvent]:
        return item.type == "tool_calls.function.arguments.delta"

    def is_end(
        self, item: ChatCompletionStreamEvent
    ) -> TypeGuard[FunctionToolCallArgumentsDoneEvent]:
        return item.type == "tool_calls.function.arguments.done"

    def transform(self, item: ChatCompletionStreamEvent) -> str:
        assert self.is_member(item)  # noqa: S101
        return item.arguments_delta

    def get_tool_name(self, item: ChatCompletionStreamEvent) -> str:
        assert self.is_member(item)  # noqa: S101
        return item.name


# TODO: Move usage tracking into OpenaiStreamState
class OpenaiUsageStreamParser(StreamParser[ChatCompletionStreamEvent, Usage]):
    """Filters and transforms OpenAI usage events from a stream."""

    def is_member(self, item: ChatCompletionStreamEvent) -> TypeGuard[ChunkEvent]:
        return item.type == "chunk" and bool(item.chunk.usage)

    def is_end(self, item: ChatCompletionStreamEvent) -> Literal[True]:
        return True  # Single event so immediately end

    def transform(self, item: ChatCompletionStreamEvent) -> Usage:
        assert self.is_member(item)  # noqa: S101
        assert item.chunk.usage  # noqa: S101
        return Usage(
            input_tokens=item.chunk.usage.prompt_tokens,
            output_tokens=item.chunk.usage.completion_tokens,
        )


class OpenaiStreamState(StreamState):
    def __init__(self, function_schemas):
        self._function_schemas = function_schemas

        self._chat_completion_stream_state = ChatCompletionStreamState(
            input_tools=openai.NOT_GIVEN,
            response_format=openai.NOT_GIVEN,
        )
        self._current_tool_call_id: str | None = None

    def update(self, item: ChatCompletionStreamEvent) -> None:
        if item.type == "chunk":
            self._chat_completion_stream_state.handle_chunk(item.chunk)
        if (
            item.type == "chunk"
            and item.chunk.choices
            and item.chunk.choices[0].delta.tool_calls
            and item.chunk.choices[0].delta.tool_calls[0].id
        ):
            # TODO: Mistral fix here ?
            # openai keeps index consistent for chunks from the same tool_call, but id is null
            # mistral has null index, but keeps id consistent
            self._current_tool_call_id = item.chunk.choices[0].delta.tool_calls[0].id

    @property
    def current_tool_call_id(self) -> str | None:
        return self._current_tool_call_id

    @property
    def current_message_snapshot(self) -> Message:
        message = (
            self._chat_completion_stream_state.current_completion_snapshot.choices[
                0
            ].message
        )
        # TODO: Possible to return AssistantMessage here?
        return _RawMessage(message.model_dump())


def _if_given(value: T | None) -> T | openai.NotGiven:
    return value if value is not None else openai.NOT_GIVEN


STR_OR_FUNCTIONCALL_TYPE = (
    str,
    StreamedStr,
    AsyncStreamedStr,
    FunctionCall,
    ParallelFunctionCall,
    AsyncParallelFunctionCall,
)

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

        match api_type:
            case "openai":
                self._client = openai.OpenAI(api_key=api_key, base_url=base_url)
                self._async_client = openai.AsyncOpenAI(
                    api_key=api_key, base_url=base_url
                )
            case "azure":
                self._client = openai.AzureOpenAI(
                    api_key=api_key,
                    base_url=base_url,  # type: ignore[arg-type]
                )
                self._async_client = openai.AsyncAzureOpenAI(
                    api_key=api_key,
                    base_url=base_url,  # type: ignore[arg-type]
                )

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

    def _get_stream_options(self) -> ChatCompletionStreamOptionsParam | openai.NotGiven:
        if self.api_type == "azure":
            return openai.NOT_GIVEN
        return {"include_usage": True}

    @staticmethod
    def _get_tool_choice(
        *,
        tool_schemas: Sequence[BaseFunctionToolSchema[Any]],
        allow_string_output: bool,
    ) -> ChatCompletionToolChoiceOptionParam | openai.NotGiven:
        """Create the tool choice argument."""
        if allow_string_output:
            return openai.NOT_GIVEN
        if len(tool_schemas) == 1:
            return tool_schemas[0].as_tool_choice()
        return "required"

    def _get_parallel_tool_calls(
        self, *, tools_specified: bool, output_types: Iterable[type]
    ) -> bool | openai.NotGiven:
        if not tools_specified:  # Enforced by OpenAI API
            return openai.NOT_GIVEN
        if self.api_type == "azure":
            return openai.NOT_GIVEN
        if is_any_origin_subclass(output_types, ParallelFunctionCall):
            return openai.NOT_GIVEN
        if is_any_origin_subclass(output_types, AsyncParallelFunctionCall):
            return openai.NOT_GIVEN
        return False

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
            output_types = cast(Iterable[type[R]], [] if functions else [str])

        # TODO: Check that Function calls types match functions
        function_schemas = [FunctionCallFunctionSchema(f) for f in functions or []] + [
            function_schema_for_type(type_)
            for type_ in output_types
            if not is_origin_subclass(type_, STR_OR_FUNCTIONCALL_TYPE)
        ]
        tool_schemas = [BaseFunctionToolSchema(schema) for schema in function_schemas]

        str_in_output_types = is_any_origin_subclass(output_types, str)
        streamed_str_in_output_types = is_any_origin_subclass(output_types, StreamedStr)
        allow_string_output = str_in_output_types or streamed_str_in_output_types

        _stream = self._client.beta.chat.completions.stream(
            model=self.model,
            messages=_add_missing_tool_calls_responses(
                [message_to_openai_message(m) for m in messages]
            ),
            max_tokens=_if_given(self.max_tokens),
            seed=_if_given(self.seed),
            stop=_if_given(stop),
            stream_options=self._get_stream_options(),
            temperature=_if_given(self.temperature),
            tools=[schema.to_dict() for schema in tool_schemas] or openai.NOT_GIVEN,
            tool_choice=self._get_tool_choice(
                tool_schemas=tool_schemas, allow_string_output=allow_string_output
            ),
            parallel_tool_calls=self._get_parallel_tool_calls(
                tools_specified=bool(tool_schemas), output_types=output_types
            ),
        ).__enter__()  # Get stream directly, without context manager
        stream = OutputStream(
            _stream,
            function_schemas=function_schemas,
            # TODO: Consoldate these into a single parser / state object?
            content_parser=OpenaiContentStreamParser(),
            tool_parser=OpenaiToolStreamParser(),
            usage_parser=OpenaiUsageStreamParser(),
            state=OpenaiStreamState(function_schemas=function_schemas),
        )
        return AssistantMessage(parse_stream(stream, output_types))  # type: ignore[return-type]

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
        tool_schemas = [BaseFunctionToolSchema(schema) for schema in function_schemas]

        str_in_output_types = is_any_origin_subclass(output_types, str)
        async_streamed_str_in_output_types = is_any_origin_subclass(
            output_types, AsyncStreamedStr
        )
        allow_string_output = str_in_output_types or async_streamed_str_in_output_types

        response: Awaitable[AsyncIterator[ChatCompletionChunk]] = (
            self._async_client.chat.completions.create(
                model=self.model,
                messages=_add_missing_tool_calls_responses(
                    [message_to_openai_message(m) for m in messages]
                ),
                max_tokens=_if_given(self.max_tokens),
                seed=_if_given(self.seed),
                stop=_if_given(stop),
                stream=True,
                stream_options=self._get_stream_options(),
                temperature=_if_given(self.temperature),
                tools=[schema.to_dict() for schema in tool_schemas] or openai.NOT_GIVEN,
                tool_choice=self._get_tool_choice(
                    tool_schemas=tool_schemas, allow_string_output=allow_string_output
                ),
                parallel_tool_calls=self._get_parallel_tool_calls(
                    tools_specified=bool(tool_schemas), output_types=output_types
                ),
            )
        )
        _stream = await AsyncChatCompletionStreamManager(
            response,
            response_format=openai.NOT_GIVEN,
            input_tools=[schema.to_dict() for schema in tool_schemas]
            or openai.NOT_GIVEN,
        ).__aenter__()  # Get stream directly, without context manager
        stream = AsyncOutputStream(
            _stream,
            function_schemas=function_schemas,
            content_parser=OpenaiContentStreamParser(),
            tool_parser=OpenaiToolStreamParser(),
            usage_parser=OpenaiUsageStreamParser(),
            state=OpenaiStreamState(function_schemas=function_schemas),
        )
        return AssistantMessage(await aparse_stream(stream, output_types))  # type: ignore[return-type]
