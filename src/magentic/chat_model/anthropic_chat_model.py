import json
from collections.abc import AsyncIterator, Callable, Iterable, Iterator
from enum import Enum
from functools import singledispatch
from typing import Any, Generic, TypeVar, cast, overload
from uuid import uuid4

from pydantic import ValidationError

from magentic import FunctionResultMessage
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
    Message,
    SystemMessage,
    UserMessage,
)
from magentic.function_call import (
    AsyncParallelFunctionCall,
    FunctionCall,
    ParallelFunctionCall,
)
from magentic.streaming import AsyncStreamedStr, StreamedStr, async_iter
from magentic.typing import is_any_origin_subclass, is_origin_subclass

try:
    import anthropic
    from anthropic.types.beta.tools import (
        ToolParam,
        ToolsBetaMessage,
        ToolsBetaMessageParam,
        ToolUseBlock,
    )
except ImportError as error:
    msg = "To use AnthropicChatModel you must install the `anthropic` package using `pip install magentic[anthropic]`."
    raise ImportError(msg) from error


class AnthropicMessageRole(Enum):
    ASSISTANT = "assistant"
    USER = "user"


@singledispatch
def message_to_anthropic_message(message: Message[Any]) -> ToolsBetaMessageParam:
    """Convert a Message to an OpenAI message."""
    # TODO: Add instructions for registering new Message type to this error message
    raise NotImplementedError(type(message))


@message_to_anthropic_message.register
def _(message: UserMessage) -> ToolsBetaMessageParam:
    return {"role": AnthropicMessageRole.USER.value, "content": message.content}


@message_to_anthropic_message.register(AssistantMessage)
def _(message: AssistantMessage[Any]) -> ToolsBetaMessageParam:
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
                # Can be random because no result will be inserted back into the chat
                "type": "tool_use",
                "id": str(uuid4()),
                "name": function_schema.name,
                "input": json.loads(function_schema.serialize_args(message.content)),
            }
        ],
    }


@message_to_anthropic_message.register(FunctionResultMessage)
def _(message: FunctionResultMessage[Any]) -> ToolsBetaMessageParam:
    function_schema = function_schema_for_type(type(message.content))
    return {
        "role": AnthropicMessageRole.USER.value,
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": message.function_call._unique_id,
                "content": json.loads(function_schema.serialize_args(message.content)),
            }
        ],
    }


T = TypeVar("T")
BaseFunctionSchemaT = TypeVar("BaseFunctionSchemaT", bound=BaseFunctionSchema[Any])


class BaseFunctionToolSchema(Generic[BaseFunctionSchemaT]):
    def __init__(self, function_schema: BaseFunctionSchemaT):
        self._function_schema = function_schema

    def to_dict(self) -> ToolParam:
        return {
            "name": self._function_schema.name,
            "description": self._function_schema.description or "",
            "input_schema": self._function_schema.parameters,  # type: ignore[typeddict-item]
        }


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
    def parse_tool_call(self, block: ToolUseBlock) -> T:
        return self._function_schema.parse_args(json.dumps(block.input))


class AsyncFunctionToolSchema(BaseFunctionToolSchema[AsyncFunctionSchema[T]]):
    async def aparse_tool_call(self, block: ToolUseBlock) -> T:
        return await self._function_schema.aparse_args(
            async_iter(json.dumps(block.input))
        )


def parse_tool_calls(
    response: ToolsBetaMessage,
    tool_schemas: Iterable[FunctionToolSchema[T]],
) -> Iterator[T]:
    for block in response.content:
        if block.type != "tool_use":
            continue
        tool_schema = select_tool_schema(block, tool_schemas)
        yield tool_schema.parse_tool_call(block)


async def aparse_tool_calls(
    response: ToolsBetaMessage,
    tool_schemas: Iterable[AsyncFunctionToolSchema[T]],
) -> AsyncIterator[T]:
    for block in response.content:
        if block.type != "tool_use":
            continue
        tool_schema = select_tool_schema(block, tool_schemas)
        yield await tool_schema.aparse_tool_call(block)


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

        response = self._client.beta.tools.messages.create(
            model=self.model,
            messages=[message_to_anthropic_message(m) for m in messages],
            max_tokens=self.max_tokens,
            # stream=True, TODO: Enable streaming when supported
            system=system,
            temperature=(
                self.temperature
                if self.temperature is not None
                else anthropic.NOT_GIVEN
            ),
            tools=[schema.to_dict() for schema in tool_schemas] or anthropic.NOT_GIVEN,
        )

        last_content = response.content[-1]

        if last_content.type == "text":
            if not allow_string_output:
                msg = (
                    "String was returned by model but not expected. You may need to update"
                    " your prompt to encourage the model to return a specific type."
                    f" {response.model_dump_json()}"
                )
                raise StructuredOutputError(msg)
            streamed_str = StreamedStr(last_content.text)
            if streamed_str_in_output_types:
                return AssistantMessage(streamed_str)  # type: ignore[return-value]
            return AssistantMessage(str(streamed_str))

        if last_content.type == "tool_use":
            try:
                if is_any_origin_subclass(output_types, ParallelFunctionCall):
                    content = ParallelFunctionCall(
                        parse_tool_calls(response, tool_schemas)
                    )
                    return AssistantMessage(content)  # type: ignore[return-value]
                # Take only the first tool_call, silently ignore extra chunks
                # TODO: Create generator here that raises error or warns if multiple tool_calls
                content = next(parse_tool_calls(response, tool_schemas))
                return AssistantMessage(content)  # type: ignore[return-value]
            except ValidationError as e:
                msg = (
                    "Failed to parse model output. You may need to update your prompt"
                    " to encourage the model to return a specific type."
                )
                raise StructuredOutputError(msg) from e

        msg = "Could not determine response type"
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

        response = await self._async_client.beta.tools.messages.create(
            model=self.model,
            messages=[message_to_anthropic_message(m) for m in messages],
            max_tokens=self.max_tokens,
            # stream=True, TODO: Enable streaming when supported
            system=system,
            temperature=(
                self.temperature
                if self.temperature is not None
                else anthropic.NOT_GIVEN
            ),
            tools=[schema.to_dict() for schema in tool_schemas] or anthropic.NOT_GIVEN,
        )

        last_content = response.content[-1]

        if last_content.type == "text":
            if not allow_string_output:
                msg = (
                    "String was returned by model but not expected. You may need to update"
                    " your prompt to encourage the model to return a specific type."
                    f" {response.model_dump_json()}"
                )
                raise StructuredOutputError(msg)
            streamed_str = AsyncStreamedStr(async_iter(last_content.text))
            if async_streamed_str_in_output_types:
                return AssistantMessage(streamed_str)  # type: ignore[return-value]
            return AssistantMessage(str(streamed_str))

        if last_content.type == "tool_use":
            try:
                if is_any_origin_subclass(output_types, AsyncParallelFunctionCall):
                    content = AsyncParallelFunctionCall(
                        aparse_tool_calls(response, tool_schemas)
                    )
                    return AssistantMessage(content)  # type: ignore[return-value]
                # Take only the first tool_call, silently ignore extra chunks
                # TODO: Create generator here that raises error or warns if multiple tool_calls
                content = await anext(aparse_tool_calls(response, tool_schemas))
                return AssistantMessage(content)  # type: ignore[return-value]
            except ValidationError as e:
                msg = (
                    "Failed to parse model output. You may need to update your prompt"
                    " to encourage the model to return a specific type."
                )
                raise StructuredOutputError(msg) from e

        msg = "Could not determine response type"
        raise ValueError(msg)
