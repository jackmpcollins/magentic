from collections.abc import AsyncIterator, Callable, Iterable
from itertools import chain
from typing import Any, TypeVar, cast, overload

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
from pydantic import ValidationError

from magentic.chat_model.base import ChatModel, StructuredOutputError
from magentic.chat_model.function_schema import (
    FunctionCallFunctionSchema,
    async_function_schema_for_type,
    function_schema_for_type,
)
from magentic.chat_model.message import (
    AssistantMessage,
    Message,
)
from magentic.chat_model.openai_chat_model import (
    AsyncFunctionToolSchema,
    BaseFunctionToolSchema,
    FunctionToolSchema,
    _aiter_streamed_tool_calls,
    _iter_streamed_tool_calls,
    message_to_openai_message,
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
    async_iter,
)
from magentic.typing import is_any_origin_subclass, is_origin_subclass

try:
    import litellm
    from litellm.utils import CustomStreamWrapper, ModelResponse
except ImportError as error:
    msg = "To use LitellmChatModel you must install the `litellm` package using `pip install magentic[litellm]`."
    raise ImportError(msg) from error


def litellm_completion(
    model: str,
    messages: list[ChatCompletionMessageParam],
    api_base: str | None = None,
    max_tokens: int | None = None,
    stop: list[str] | None = None,
    temperature: float | None = None,
    tools: list[ChatCompletionToolParam] | None = None,
    tool_choice: ChatCompletionToolChoiceOptionParam | None = None,
) -> CustomStreamWrapper:
    """Type-annotated version of `litellm.completion`."""
    # `litellm.completion` doesn't accept `None`
    # so only pass args with values
    kwargs: dict[str, Any] = {}
    if api_base is not None:
        kwargs["api_base"] = api_base
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if temperature is not None:
        kwargs["temperature"] = temperature
    if tools:
        kwargs["tools"] = tools
    if tool_choice:
        kwargs["tool_choice"] = tool_choice

    response: CustomStreamWrapper = litellm.completion(  # type: ignore[no-untyped-call,unused-ignore]
        model=model,
        messages=messages,
        stop=stop,
        stream=True,
        **kwargs,
    )
    return response


async def litellm_acompletion(
    model: str,
    messages: list[ChatCompletionMessageParam],
    api_base: str | None = None,
    max_tokens: int | None = None,
    stop: list[str] | None = None,
    temperature: float | None = None,
    tools: list[ChatCompletionToolParam] | None = None,
    tool_choice: ChatCompletionToolChoiceOptionParam | None = None,
) -> AsyncIterator[ModelResponse]:
    """Type-annotated version of `litellm.acompletion`."""
    # `litellm.acompletion` doesn't accept `None`
    # so only pass args with values
    kwargs: dict[str, Any] = {}
    if api_base is not None:
        kwargs["api_base"] = api_base
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if temperature is not None:
        kwargs["temperature"] = temperature
    if tools:
        kwargs["tools"] = tools
    if tool_choice:
        kwargs["tool_choice"] = tool_choice

    response: AsyncIterator[ModelResponse] = await litellm.acompletion(  # type: ignore[no-untyped-call,unused-ignore]
        model=model,
        messages=messages,
        stop=stop,
        stream=True,
        **kwargs,
    )
    return response


BeseToolSchemaT = TypeVar("BeseToolSchemaT", bound=BaseFunctionToolSchema[Any])
R = TypeVar("R")


class LitellmChatModel(ChatModel):
    """An LLM chat model that uses the `litellm` python package."""

    def __init__(
        self,
        model: str,
        *,
        api_base: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ):
        self._model = model
        self._api_base = api_base
        self._max_tokens = max_tokens
        self._temperature = temperature

    @property
    def model(self) -> str:
        return self._model

    @property
    def api_base(self) -> str | None:
        return self._api_base

    @property
    def max_tokens(self) -> int | None:
        return self._max_tokens

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

        response = litellm_completion(
            model=self.model,
            messages=[message_to_openai_message(m) for m in messages],
            api_base=self.api_base,
            max_tokens=self.max_tokens,
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
                chunk.choices[0].delta.get("content", None)
                for chunk in response
                if chunk.choices[0].delta.get("content", None) is not None
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
    ) -> AssistantMessage[str] | AssistantMessage[R]:
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

        response = await litellm_acompletion(
            model=self.model,
            messages=[message_to_openai_message(m) for m in messages],
            api_base=self.api_base,
            max_tokens=self.max_tokens,
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
                chunk.choices[0].delta.get("content", None)
                async for chunk in response
                if chunk.choices[0].delta.get("content", None) is not None
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
