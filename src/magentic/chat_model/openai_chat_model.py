from collections.abc import AsyncIterable, AsyncIterator, Callable, Iterable, Iterator
from enum import Enum
from functools import singledispatch, wraps
from itertools import chain, groupby
from typing import Any, Generic, Literal, ParamSpec, Sequence, TypeVar, cast, overload

import openai
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionStreamOptionsParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
from pydantic import ValidationError

from magentic.chat_model.base import (
    ChatModel,
    StructuredOutputError,
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
    agroupby,
    async_iter,
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
    function_schema = function_schema_for_type(type(message.content))
    return {
        "role": OpenaiMessageRole.TOOL.value,
        "tool_call_id": message.tool_call_id,
        "content": function_schema.serialize_args(message.content),
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

    def matches(self, tool_call: ChoiceDeltaToolCall) -> bool:
        return bool(
            # TODO: Add back tool_call.type == "function" when LiteLLM Mistral fixed
            # https://github.com/BerriAI/litellm/issues/2645
            tool_call.function and self._function_schema.name == tool_call.function.name
        )


# TODO: Generalize this to BaseToolSchema when that is created
BeseToolSchemaT = TypeVar("BeseToolSchemaT", bound=BaseFunctionToolSchema[Any])


def select_tool_schema(
    tool_call: ChoiceDeltaToolCall, tools_schemas: list[BeseToolSchemaT]
) -> BeseToolSchemaT:
    """Select the tool schema based on the response chunk."""
    for tool_schema in tools_schemas:
        if tool_schema.matches(tool_call):
            return tool_schema

    msg = f"Unknown tool call: {tool_call.model_dump_json()}"
    raise ValueError(msg)


class FunctionToolSchema(BaseFunctionToolSchema[FunctionSchema[T]]):
    def parse_tool_call(self, chunks: Iterable[ChoiceDeltaToolCall]) -> T:
        return self._function_schema.parse_args(
            chunk.function.arguments
            for chunk in chunks
            if chunk.function and chunk.function.arguments is not None
        )


class AsyncFunctionToolSchema(BaseFunctionToolSchema[AsyncFunctionSchema[T]]):
    async def aparse_tool_call(self, chunks: AsyncIterable[ChoiceDeltaToolCall]) -> T:
        return await self._function_schema.aparse_args(
            chunk.function.arguments
            async for chunk in chunks
            if chunk.function and chunk.function.arguments is not None
        )


def _get_tool_call_id_for_chunk(tool_call: ChoiceDeltaToolCall) -> Any:
    """Returns an id that is consistent for chunks from the same tool_call."""
    # openai keeps index consistent for chunks from the same tool_call, but id is null
    # mistral has null index, but keeps id consistent
    return tool_call.index if tool_call.index is not None else tool_call.id


def parse_streamed_tool_calls(
    response: Iterable[ChatCompletionChunk],
    tool_schemas: list[FunctionToolSchema[T]],
) -> Iterator[T]:
    all_tool_call_chunks = (
        tool_call
        for chunk in response
        if chunk.choices and chunk.choices[0].delta.tool_calls
        for tool_call in chunk.choices[0].delta.tool_calls
    )
    for _, tool_call_chunks in groupby(
        all_tool_call_chunks, _get_tool_call_id_for_chunk
    ):
        first_chunk = next(tool_call_chunks)
        tool_schema = select_tool_schema(first_chunk, tool_schemas)
        tool_call = tool_schema.parse_tool_call(chain([first_chunk], tool_call_chunks))  # noqa: B031
        yield tool_call


async def aparse_streamed_tool_calls(
    response: AsyncIterable[ChatCompletionChunk],
    tool_schemas: list[AsyncFunctionToolSchema[T]],
) -> AsyncIterator[T]:
    all_tool_call_chunks = (
        tool_call
        async for chunk in response
        if chunk.choices and chunk.choices[0].delta.tool_calls
        for tool_call in chunk.choices[0].delta.tool_calls
    )
    async for _, tool_call_chunks in agroupby(
        all_tool_call_chunks, _get_tool_call_id_for_chunk
    ):
        first_chunk = await anext(tool_call_chunks)
        tool_schema = select_tool_schema(first_chunk, tool_schemas)
        tool_call = await tool_schema.aparse_tool_call(
            achain(async_iter([first_chunk]), tool_call_chunks)
        )
        yield tool_call


def _create_usage_ref(
    response: Iterable[ChatCompletionChunk],
) -> tuple[list[Usage], Iterator[ChatCompletionChunk]]:
    """Returns a pointer to a Usage object that is created at the end of the response."""
    usage_ref: list[Usage] = []

    def generator(
        response: Iterable[ChatCompletionChunk],
    ) -> Iterator[ChatCompletionChunk]:
        for chunk in response:
            if chunk.usage:
                usage = Usage(
                    input_tokens=chunk.usage.prompt_tokens,
                    output_tokens=chunk.usage.completion_tokens,
                )
                usage_ref.append(usage)
            yield chunk

    return usage_ref, generator(response)


def _create_usage_ref_async(
    response: AsyncIterable[ChatCompletionChunk],
) -> tuple[list[Usage], AsyncIterator[ChatCompletionChunk]]:
    """Async version of `_create_usage_ref`."""
    usage_ref: list[Usage] = []

    async def agenerator(
        response: AsyncIterable[ChatCompletionChunk],
    ) -> AsyncIterator[ChatCompletionChunk]:
        async for chunk in response:
            if chunk.usage:
                usage = Usage(
                    input_tokens=chunk.usage.prompt_tokens,
                    output_tokens=chunk.usage.completion_tokens,
                )
                usage_ref.append(usage)
            yield chunk

    return usage_ref, agenerator(response)


P = ParamSpec("P")
R = TypeVar("R")


def discard_none_arguments(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator to discard function arguments with value `None`"""

    @wraps(func)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        non_none_kwargs = {
            key: value for key, value in kwargs.items() if value is not None
        }
        return func(*args, **non_none_kwargs)  # type: ignore[arg-type]

    return wrapped


STR_OR_FUNCTIONCALL_TYPE = (
    str,
    StreamedStr,
    AsyncStreamedStr,
    FunctionCall,
    ParallelFunctionCall,
    AsyncParallelFunctionCall,
)


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
        tool_schemas = [FunctionToolSchema(schema) for schema in function_schemas]

        str_in_output_types = is_any_origin_subclass(output_types, str)
        streamed_str_in_output_types = is_any_origin_subclass(output_types, StreamedStr)
        allow_string_output = str_in_output_types or streamed_str_in_output_types

        response: Iterator[ChatCompletionChunk] = discard_none_arguments(
            self._client.chat.completions.create
        )(
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
        usage_ref, response = _create_usage_ref(response)

        first_chunk = next(response)
        # Azure OpenAI sends a chunk with empty choices first
        if len(first_chunk.choices) == 0:
            first_chunk = next(response)
        if (
            # Mistral tool call first chunk has content ""
            not first_chunk.choices[0].delta.content
            and not first_chunk.choices[0].delta.tool_calls
        ):
            first_chunk = next(response)
        response = chain([first_chunk], response)

        if first_chunk.choices[0].delta.content:
            streamed_str = StreamedStr(
                chunk.choices[0].delta.content
                for chunk in response
                if chunk.choices and chunk.choices[0].delta.content is not None
            )
            str_content = validate_str_content(
                streamed_str,
                allow_string_output=allow_string_output,
                streamed=streamed_str_in_output_types,
            )
            return AssistantMessage._with_usage(str_content, usage_ref)  # type: ignore[return-value]

        if first_chunk.choices[0].delta.tool_calls:
            try:
                if is_any_origin_subclass(output_types, ParallelFunctionCall):
                    content = ParallelFunctionCall(
                        parse_streamed_tool_calls(response, tool_schemas)
                    )
                    return AssistantMessage._with_usage(content, usage_ref)  # type: ignore[return-value]
                # Take only the first tool_call, silently ignore extra chunks
                # TODO: Create generator here that raises error or warns if multiple tool_calls
                content = next(parse_streamed_tool_calls(response, tool_schemas))
                return AssistantMessage._with_usage(content, usage_ref)  # type: ignore[return-value]
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
            if not is_origin_subclass(type_, STR_OR_FUNCTIONCALL_TYPE)
        ]
        tool_schemas = [AsyncFunctionToolSchema(schema) for schema in function_schemas]

        str_in_output_types = is_any_origin_subclass(output_types, str)
        async_streamed_str_in_output_types = is_any_origin_subclass(
            output_types, AsyncStreamedStr
        )
        allow_string_output = str_in_output_types or async_streamed_str_in_output_types

        response: AsyncIterator[ChatCompletionChunk] = await discard_none_arguments(
            self._async_client.chat.completions.create
        )(
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
        usage_ref, response = _create_usage_ref_async(response)

        first_chunk = await anext(response)
        # Azure OpenAI sends a chunk with empty choices first
        if len(first_chunk.choices) == 0:
            first_chunk = await anext(response)
        if (
            # Mistral tool call first chunk has content ""
            not first_chunk.choices[0].delta.content
            and not first_chunk.choices[0].delta.tool_calls
        ):
            first_chunk = await anext(response)
        response = achain(async_iter([first_chunk]), response)

        if first_chunk.choices[0].delta.content:
            async_streamed_str = AsyncStreamedStr(
                chunk.choices[0].delta.content
                async for chunk in response
                if chunk.choices and chunk.choices[0].delta.content is not None
            )
            str_content = await avalidate_str_content(
                async_streamed_str,
                allow_string_output=allow_string_output,
                streamed=async_streamed_str_in_output_types,
            )
            return AssistantMessage._with_usage(str_content, usage_ref)  # type: ignore[return-value]

        if first_chunk.choices[0].delta.tool_calls:
            try:
                if is_any_origin_subclass(output_types, AsyncParallelFunctionCall):
                    content = AsyncParallelFunctionCall(
                        aparse_streamed_tool_calls(response, tool_schemas)
                    )
                    return AssistantMessage._with_usage(content, usage_ref)  # type: ignore[return-value]

                # Take only the first tool_call, silently ignore extra chunks
                content = await anext(
                    aparse_streamed_tool_calls(response, tool_schemas)
                )
                return AssistantMessage._with_usage(content, usage_ref)  # type: ignore[return-value]
            except ValidationError as e:
                msg = (
                    "Failed to parse model output. You may need to update your prompt"
                    " to encourage the model to return a specific type."
                )
                raise StructuredOutputError(msg) from e

        msg = f"Could not determine response type for first chunk: {first_chunk.model_dump_json()}"
        raise ValueError(msg)
