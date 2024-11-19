import base64
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Iterator,
    Sequence,
)
from enum import Enum
from functools import singledispatch
from itertools import chain
from typing import Any, Generic, Literal, TypeVar, cast, overload

import filetype
import openai
from openai.lib.streaming.chat import (
    AsyncChatCompletionStream,
    AsyncChatCompletionStreamManager,
    ChatCompletionStream,
)
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionStreamOptionsParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
)
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
    achain,
    async_iter,
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


class OpenaiStream:
    """Converts a stream of openai events into a stream of magentic objects."""

    def __init__(
        self, stream: ChatCompletionStream, function_schemas: list[FunctionSchema[Any]]
    ):
        self._stream = stream
        self._function_schemas = function_schemas
        self._iterator = self.__stream__()
        self.usage: Usage | None = None

    def __next__(self) -> StreamedStr | FunctionCall:
        return self._iterator.__next__()

    def __iter__(self) -> Iterator[StreamedStr | FunctionCall]:
        yield from self._iterator

    def __stream__(self) -> Iterator[StreamedStr | FunctionCall]:
        transition = [next(self._stream)]

        def _streamed_str(stream: Iterator) -> StreamedStr:
            def _group(stream: Iterator) -> Iterator:
                for event in stream:
                    if event.type == "content.delta":
                        yield event.delta
                    elif event.type == "content.done":
                        transition.append(event)
                        return

            return StreamedStr(_group(stream))

        def _function_call(transition_item, stream: Iterator) -> FunctionCall:
            def _group(stream: Iterator) -> Iterator:
                for event in stream:
                    if event.type == "tool_calls.function.arguments.delta":
                        yield event.arguments_delta
                    elif event.type == "tool_calls.function.arguments.done":
                        transition.append(event)
                        return

            # TODO: Tidy matching function schema. Include Mistral fix
            for function_schema in self._function_schemas:
                if function_schema.name == transition_item.name:
                    break
            # TODO: Catch/raise unknown tool call error here
            try:  # TODO: Tidy catching of error here to DRY with async
                return function_schema.parse_args(_group(stream))
            except ValidationError as e:
                raw_message = self._stream.current_completion_snapshot.choices[
                    0
                ].message.model_dump()
                raise ToolSchemaParseError(
                    output_message=_RawMessage(raw_message),
                    tool_call_id=raw_message.content["tool_calls"][0]["id"],  # type: ignore[index,unused-ignore]
                    validation_error=e,
                ) from e

        while transition:
            transition_item = transition.pop()
            if transition_item.type == "content.delta":
                yield _streamed_str(self._stream)
            elif transition_item.type == "tool_calls.function.arguments.delta":
                yield _function_call(transition_item, self._stream)
            elif transition_item.type == "chunk" and transition_item.chunk.usage:
                self.usage = Usage(
                    input_tokens=transition_item.chunk.usage.prompt_tokens,
                    output_tokens=transition_item.chunk.usage.completion_tokens,
                )
            elif new_transition_item := next(self._stream, None):
                transition.append(new_transition_item)

    def close(self):
        self._stream.close()


class OpenaiAsyncStream:
    """Converts an async stream of openai events into an async stream of magentic objects."""

    def __init__(
        self,
        stream: AsyncChatCompletionStream,
        function_schemas: list[AsyncFunctionSchema[Any]],
    ):
        self._stream = stream
        self._function_schemas = function_schemas
        self._aiterator = self.__stream__()
        self.usage: Usage | None = None

    async def __anext__(self) -> AsyncStreamedStr | FunctionCall:
        return await self._aiterator.__anext__()

    async def __aiter__(self) -> AsyncIterator[AsyncStreamedStr | FunctionCall]:
        async for item in self._aiterator:
            yield item

    async def __stream__(self) -> AsyncIterator[AsyncStreamedStr | FunctionCall]:
        transition = [await anext(self._stream)]

        def _streamed_str(stream: AsyncIterator) -> AsyncStreamedStr:
            async def _group(stream: AsyncIterator) -> AsyncIterator:
                async for event in stream:
                    if event.type == "content.delta":
                        yield event.delta
                    elif event.type == "content.done":
                        transition.append(event)
                        return

            return AsyncStreamedStr(_group(stream))

        async def _function_call(
            transition_item, stream: AsyncIterator
        ) -> FunctionCall:
            async def _group(stream: AsyncIterator) -> AsyncIterator:
                async for event in stream:
                    if event.type == "tool_calls.function.arguments.delta":
                        yield event.arguments_delta
                    elif event.type == "tool_calls.function.arguments.done":
                        transition.append(event)
                        return

            # TODO: Tidy matching function schema. Include Mistral fix
            for function_schema in self._function_schemas:
                if function_schema.name == transition_item.name:
                    break
            # TODO: Catch/raise unknown tool call error here
            try:  # TODO: Tidy catching of error here to DRY with async
                return await function_schema.aparse_args(_group(stream))
            except ValidationError as e:
                raw_message = self._stream.current_completion_snapshot.choices[
                    0
                ].message.model_dump()
                raise ToolSchemaParseError(
                    output_message=_RawMessage(raw_message),
                    tool_call_id=raw_message.content["tool_calls"][0]["id"],  # type: ignore[index,unused-ignore]
                    validation_error=e,
                ) from e

        while transition:
            transition_item = transition.pop()
            if transition_item.type == "content.delta":
                yield _streamed_str(self._stream)
            elif transition_item.type == "tool_calls.function.arguments.delta":
                yield await _function_call(transition_item, self._stream)
            elif transition_item.type == "chunk" and transition_item.chunk.usage:
                self.usage = Usage(
                    input_tokens=transition_item.chunk.usage.prompt_tokens,
                    output_tokens=transition_item.chunk.usage.completion_tokens,
                )
            elif new_transition_item := await anext(self._stream, None):
                transition.append(new_transition_item)

    async def close(self):
        await self._stream.close()


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
            output_types = [] if functions else cast(list[type[R]], [str])

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
            max_tokens=self.max_tokens,
            seed=self.seed,
            stop=stop,
            stream_options=self._get_stream_options(),
            temperature=self.temperature,
            tools=[schema.to_dict() for schema in tool_schemas] or openai.NOT_GIVEN,
            tool_choice=self._get_tool_choice(
                tool_schemas=tool_schemas, allow_string_output=allow_string_output
            ),
            parallel_tool_calls=self._get_parallel_tool_calls(
                tools_specified=bool(tool_schemas), output_types=output_types
            ),
        ).__enter__()  # Get stream directly, without context manager
        stream = OpenaiStream(_stream, function_schemas=function_schemas)

        # TODO: Function to validate LLM output against prompt-function return type
        first_response_obj = next(stream)
        if isinstance(first_response_obj, StreamedStr):
            str_content = validate_str_content(
                first_response_obj,
                allow_string_output=allow_string_output,
                streamed=streamed_str_in_output_types,
            )
            return AssistantMessage(str_content)  # type: ignore[return-value]

        if isinstance(first_response_obj, FunctionCall):
            if is_any_origin_subclass(output_types, ParallelFunctionCall):
                content = ParallelFunctionCall(chain([first_response_obj], stream))
                return AssistantMessage(content)  # type: ignore[return-value]
            # Take only the first tool_call, silently ignore extra chunks
            return AssistantMessage(first_response_obj)  # type: ignore[return-value]

        return AssistantMessage(first_response_obj)

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
                max_tokens=self.max_tokens,
                seed=self.seed,
                stop=stop,
                stream=True,
                stream_options=self._get_stream_options(),
                temperature=self.temperature,
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
        stream = OpenaiAsyncStream(_stream, function_schemas=function_schemas)

        # TODO: Function to validate LLM output against prompt-function return type
        first_response_obj = await anext(stream)
        if isinstance(first_response_obj, AsyncStreamedStr):
            str_content = await avalidate_str_content(
                first_response_obj,
                allow_string_output=allow_string_output,
                streamed=async_streamed_str_in_output_types,
            )
            return AssistantMessage(str_content)  # type: ignore[return-value]

        if isinstance(first_response_obj, FunctionCall):
            if is_any_origin_subclass(output_types, AsyncParallelFunctionCall):
                content = AsyncParallelFunctionCall(
                    achain(async_iter([first_response_obj]), stream)
                )
                return AssistantMessage(content)  # type: ignore[return-value]
            # Take only the first tool_call, silently ignore extra chunks
            return AssistantMessage(first_response_obj)  # type: ignore[return-value]

        return AssistantMessage(first_response_obj)
