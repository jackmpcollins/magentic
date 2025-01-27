from collections.abc import AsyncIterator, Callable, Iterable, Iterator, Sequence
from enum import Enum
from functools import singledispatch
from typing import Any, Generic, Literal, TypeVar, cast, overload

import openai
from openai.lib.streaming.chat import ChatCompletionStreamState
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionContentPartParam,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionStreamOptionsParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)

from magentic._parsing import contains_parallel_function_call_type, contains_string_type
from magentic._streamed_response import AsyncStreamedResponse, StreamedResponse
from magentic.chat_model.base import ChatModel, aparse_stream, parse_stream
from magentic.chat_model.function_schema import (
    BaseFunctionSchema,
    FunctionCallFunctionSchema,
    function_schema_for_type,
    get_async_function_schemas,
    get_function_schemas,
)
from magentic.chat_model.message import (
    AssistantMessage,
    ImageBytes,
    ImageUrl,
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


@singledispatch
async def async_message_to_openai_message(
    message: Message[Any],
) -> ChatCompletionMessageParam:
    """Async version of `message_to_openai_message`."""
    return message_to_openai_message(message)


@message_to_openai_message.register(_RawMessage)
def _(message: _RawMessage[Any]) -> ChatCompletionMessageParam:
    assert isinstance(message.content, dict)
    assert "role" in message.content
    assert "content" in message.content
    return cast(ChatCompletionMessageParam, message.content)


@message_to_openai_message.register
def _(message: SystemMessage) -> ChatCompletionMessageParam:
    return {"role": OpenaiMessageRole.SYSTEM.value, "content": message.content}


@message_to_openai_message.register(UserMessage)
def _(message: UserMessage[Any]) -> ChatCompletionUserMessageParam:
    if isinstance(message.content, str):
        return {"role": OpenaiMessageRole.USER.value, "content": message.content}
    if isinstance(message.content, Iterable):
        content: list[ChatCompletionContentPartParam] = []
        for block in message.content:
            if isinstance(block, str):
                content.append({"type": "text", "text": block})
            elif isinstance(block, ImageBytes):
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{block.mime_type};base64,{block.as_base64()}"
                        },
                    }
                )
            elif isinstance(block, ImageUrl):
                content.append({"type": "image_url", "image_url": {"url": block.root}})
            else:
                msg = f"Invalid block type: {type(block)}"
                raise TypeError(msg)
        return {"role": OpenaiMessageRole.USER.value, "content": content}
    msg = f"Invalid content type: {type(message.content)}"
    raise TypeError(msg)


@message_to_openai_message.register(UserImageMessage)
def _(message: UserImageMessage[Any]) -> ChatCompletionUserMessageParam:
    if isinstance(message.content, bytes):
        image_bytes = ImageBytes(message.content)
        url = f"data:{image_bytes.mime_type};base64,{image_bytes.as_base64()}"
    elif isinstance(message.content, str):
        url = message.content
    else:
        msg = f"Invalid content type: {type(message.content)}"
        raise TypeError(msg)

    return {
        "role": OpenaiMessageRole.USER.value,
        "content": [{"type": "image_url", "image_url": {"url": url, "detail": "auto"}}],
    }


def _function_call_to_tool_call_block(
    function_call: FunctionCall[Any],
) -> ChatCompletionMessageToolCallParam:
    function_schema = FunctionCallFunctionSchema(function_call.function)
    return {
        "id": function_call._unique_id,
        "type": "function",
        "function": {
            "name": function_schema.name,
            "arguments": function_schema.serialize_args(function_call),
        },
    }


@message_to_openai_message.register(AssistantMessage)
def _(message: AssistantMessage[Any]) -> ChatCompletionMessageParam:
    if isinstance(message.content, str):
        return {"role": OpenaiMessageRole.ASSISTANT.value, "content": message.content}

    if isinstance(message.content, FunctionCall):
        return {
            "role": OpenaiMessageRole.ASSISTANT.value,
            "tool_calls": [_function_call_to_tool_call_block(message.content)],
        }

    if isinstance(message.content, ParallelFunctionCall):
        return {
            "role": OpenaiMessageRole.ASSISTANT.value,
            "tool_calls": [
                _function_call_to_tool_call_block(function_call)
                for function_call in message.content
            ],
        }

    if isinstance(message.content, StreamedResponse):
        content: list[str] = []
        function_calls: list[FunctionCall[Any]] = []
        for item in message.content:
            if isinstance(item, StreamedStr):
                content.append(item.to_string())
            elif isinstance(item, FunctionCall):
                function_calls.append(item)
        return {
            "role": OpenaiMessageRole.ASSISTANT.value,
            "content": " ".join(content),
            "tool_calls": [
                _function_call_to_tool_call_block(function_call)
                for function_call in function_calls
            ],
        }

    function_schema = function_schema_for_type(type(message.content))
    return {
        "role": OpenaiMessageRole.ASSISTANT.value,
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


@async_message_to_openai_message.register(AssistantMessage)
async def _(message: AssistantMessage[Any]) -> ChatCompletionMessageParam:
    if isinstance(message.content, AsyncParallelFunctionCall):
        return {
            "role": OpenaiMessageRole.ASSISTANT.value,
            "tool_calls": [
                _function_call_to_tool_call_block(function_call)
                async for function_call in message.content
            ],
        }

    if isinstance(message.content, AsyncStreamedResponse):
        content: list[str] = []
        function_calls: list[FunctionCall[Any]] = []
        async for item in message.content:
            if isinstance(item, AsyncStreamedStr):
                content.append(await item.to_string())
            elif isinstance(item, FunctionCall):
                function_calls.append(item)
        return {
            "role": OpenaiMessageRole.ASSISTANT.value,
            "content": " ".join(content),
            "tool_calls": [
                _function_call_to_tool_call_block(function_call)
                for function_call in function_calls
            ],
        }
    return message_to_openai_message(message)


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

    def as_tool_choice(self) -> ChatCompletionNamedToolChoiceParam:
        return {"type": "function", "function": {"name": self._function_schema.name}}

    def to_dict(self) -> ChatCompletionToolParam:
        return {"type": "function", "function": self._function_schema.dict()}


class OpenaiStreamParser(StreamParser[ChatCompletionChunk]):
    def is_content(self, item: ChatCompletionChunk) -> bool:
        return bool(item.choices and item.choices[0].delta.content)

    def get_content(self, item: ChatCompletionChunk) -> str | None:
        if item.choices and item.choices[0].delta.content:
            return item.choices[0].delta.content
        return None

    def is_tool_call(self, item: ChatCompletionChunk) -> bool:
        return bool(item.choices and item.choices[0].delta.tool_calls)

    def iter_tool_calls(self, item: ChatCompletionChunk) -> Iterator[FunctionCallChunk]:
        if item.choices and item.choices[0].delta.tool_calls:
            for tool_call in item.choices[0].delta.tool_calls:
                if tool_call.function:
                    yield FunctionCallChunk(
                        id=tool_call.id,
                        name=tool_call.function.name,
                        args=tool_call.function.arguments,
                    )


class OpenaiStreamState(StreamState[ChatCompletionChunk]):
    """Tracks the state of the OpenAI model output stream.

    - message snapshot
    - usage
    - stop reason
    """

    def __init__(self) -> None:
        self._chat_completion_stream_state = ChatCompletionStreamState(
            input_tools=openai.NOT_GIVEN,
            response_format=openai.NOT_GIVEN,
        )
        self.usage_ref: list[Usage] = []

        # Keep track of tool call index to add this to Mistral tool calls
        self._current_tool_call_index: int = -1
        self._seen_tool_call_ids: set[str] = set()

    def update(self, item: ChatCompletionChunk) -> None:
        # Add tool call index for Mistral tool calls to make compatible with OpenAI
        # TODO: Remove this fix when MistralChatModel switched to mistral python package
        if item.choices:
            for tool_call_chunk in item.choices[0].delta.tool_calls or []:
                if (
                    tool_call_chunk.id is not None
                    and tool_call_chunk.id not in self._seen_tool_call_ids
                ):
                    self._current_tool_call_index += 1
                    self._seen_tool_call_ids.add(tool_call_chunk.id)
                tool_call_chunk.index = self._current_tool_call_index
        self._chat_completion_stream_state.handle_chunk(item)
        if item.usage:
            assert not self.usage_ref
            self.usage_ref.append(
                Usage(
                    input_tokens=item.usage.prompt_tokens,
                    output_tokens=item.usage.completion_tokens,
                )
            )

    @property
    def current_message_snapshot(self) -> Message[Any]:
        snapshot = self._chat_completion_stream_state.current_completion_snapshot
        message = snapshot.choices[0].message
        # TODO: Possible to return AssistantMessage here?
        return _RawMessage(message.model_dump())


def _if_given(value: T | None) -> T | openai.NotGiven:
    return value if value is not None else openai.NOT_GIVEN


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
        output_types: Iterable[type],
    ) -> ChatCompletionToolChoiceOptionParam | openai.NotGiven:
        """Create the tool choice argument."""
        if contains_string_type(output_types):
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
        if contains_parallel_function_call_type(output_types):
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
        # TODO: Add type hint for function call ?
    ) -> AssistantMessage[str] | AssistantMessage[R]:
        """Request an LLM message."""
        if output_types is None:
            output_types = cast(Iterable[type[R]], [] if functions else [str])

        function_schemas = get_function_schemas(functions, output_types)
        tool_schemas = [BaseFunctionToolSchema(schema) for schema in function_schemas]

        response: Iterator[ChatCompletionChunk] = self._client.chat.completions.create(
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
                tool_schemas=tool_schemas, output_types=output_types
            ),
            parallel_tool_calls=self._get_parallel_tool_calls(
                tools_specified=bool(tool_schemas), output_types=output_types
            ),
        )
        stream = OutputStream(
            response,
            function_schemas=function_schemas,
            parser=OpenaiStreamParser(),
            state=OpenaiStreamState(),
        )
        return AssistantMessage._with_usage(
            parse_stream(stream, output_types), usage_ref=stream.usage_ref
        )

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

        function_schemas = get_async_function_schemas(functions, output_types)
        tool_schemas = [BaseFunctionToolSchema(schema) for schema in function_schemas]

        response: AsyncIterator[
            ChatCompletionChunk
        ] = await self._async_client.chat.completions.create(
            model=self.model,
            messages=_add_missing_tool_calls_responses(
                [await async_message_to_openai_message(m) for m in messages]
            ),
            max_tokens=_if_given(self.max_tokens),
            seed=_if_given(self.seed),
            stop=_if_given(stop),
            stream=True,
            stream_options=self._get_stream_options(),
            temperature=_if_given(self.temperature),
            tools=[schema.to_dict() for schema in tool_schemas] or openai.NOT_GIVEN,
            tool_choice=self._get_tool_choice(
                tool_schemas=tool_schemas, output_types=output_types
            ),
            parallel_tool_calls=self._get_parallel_tool_calls(
                tools_specified=bool(tool_schemas), output_types=output_types
            ),
        )
        stream = AsyncOutputStream(
            response,
            function_schemas=function_schemas,
            parser=OpenaiStreamParser(),
            state=OpenaiStreamState(),
        )
        return AssistantMessage._with_usage(
            await aparse_stream(stream, output_types), usage_ref=stream.usage_ref
        )
