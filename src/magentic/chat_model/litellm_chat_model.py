from collections.abc import AsyncIterator, Callable, Iterable
from itertools import chain
from typing import Any, Literal, TypeVar, cast, overload

from litellm.utils import CustomStreamWrapper, ModelResponse
from openai.types.chat import ChatCompletionMessageParam

from magentic.chat_model.openai_chat_model import message_to_openai_message

try:
    import litellm
except ImportError as error:
    msg = "To use LitellmChatModel you must install the `litellm` package."
    raise ImportError(msg) from error
from pydantic import ValidationError

from magentic.chat_model.base import ChatModel, StructuredOutputError
from magentic.chat_model.function_schema import (
    BaseFunctionSchema,
    FunctionCallFunctionSchema,
    async_function_schema_for_type,
    function_schema_for_type,
)
from magentic.chat_model.message import (
    AssistantMessage,
    Message,
)
from magentic.function_call import FunctionCall
from magentic.streaming import (
    AsyncStreamedStr,
    StreamedStr,
    achain,
    async_iter,
)
from magentic.typing import is_origin_subclass


def litellm_completion(
    model: str,
    messages: list[ChatCompletionMessageParam],
    api_base: str | None = None,
    max_tokens: int | None = None,
    stop: list[str] | None = None,
    temperature: float | None = None,
    functions: list[dict[str, Any]] | None = None,
    function_call: Literal["auto", "none"] | dict[str, Any] | None = None,
) -> CustomStreamWrapper:
    """Type-annotated version of `litellm.completion`."""
    # `litellm.completion` doesn't accept `None`
    # so only pass args with values
    kwargs: dict[str, Any] = {}
    if api_base is not None:
        kwargs["api_base"] = api_base
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if functions:
        kwargs["functions"] = functions
    if function_call:
        kwargs["function_call"] = function_call
    if temperature is not None:
        kwargs["temperature"] = temperature

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
    functions: list[dict[str, Any]] | None = None,
    function_call: Literal["auto", "none"] | dict[str, Any] | None = None,
) -> AsyncIterator[ModelResponse]:
    """Type-annotated version of `litellm.acompletion`."""
    # `litellm.acompletion` doesn't accept `None`
    # so only pass args with values
    kwargs: dict[str, Any] = {}
    if api_base is not None:
        kwargs["api_base"] = api_base
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if functions:
        kwargs["functions"] = functions
    if function_call:
        kwargs["function_call"] = function_call
    if temperature is not None:
        kwargs["temperature"] = temperature

    response: AsyncIterator[ModelResponse] = await litellm.acompletion(  # type: ignore[no-untyped-call,unused-ignore]
        model=model,
        messages=messages,
        stop=stop,
        stream=True,
        **kwargs,
    )
    return response


BeseFunctionSchemaT = TypeVar("BeseFunctionSchemaT", bound=BaseFunctionSchema[Any])
R = TypeVar("R")
FuncR = TypeVar("FuncR")


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
    def _select_function_schema(
        chunk: ModelResponse, function_schemas: list[BeseFunctionSchemaT]
    ) -> BeseFunctionSchemaT | None:
        """Select the function schema based on the first response chunk."""
        if not chunk.choices[0].delta.get("function_call", None):
            return None

        function_name: str | None = chunk.choices[0].delta.function_call.name
        if function_name is None:
            msg = f"LiteLLM function call name is None. Chunk: {chunk.json()}"
            raise ValueError(msg)

        function_schema_by_name = {
            function_schema.name: function_schema
            for function_schema in function_schemas
        }
        return function_schema_by_name[function_name]

    @overload
    def complete(
        self,
        messages: Iterable[Message[Any]],
        functions: None = ...,
        output_types: None = ...,
        *,
        stop: list[str] | None = ...,
    ) -> AssistantMessage[str]:
        ...

    @overload
    def complete(
        self,
        messages: Iterable[Message[Any]],
        functions: Iterable[Callable[..., FuncR]],
        output_types: None = ...,
        *,
        stop: list[str] | None = ...,
    ) -> AssistantMessage[FunctionCall[FuncR]] | AssistantMessage[str]:
        ...

    @overload
    def complete(
        self,
        messages: Iterable[Message[Any]],
        functions: None = ...,
        output_types: Iterable[type[R]] = ...,
        *,
        stop: list[str] | None = ...,
    ) -> AssistantMessage[R]:
        ...

    @overload
    def complete(
        self,
        messages: Iterable[Message[Any]],
        functions: Iterable[Callable[..., FuncR]],
        output_types: Iterable[type[R]],
        *,
        stop: list[str] | None = ...,
    ) -> AssistantMessage[FunctionCall[FuncR]] | AssistantMessage[R]:
        ...

    def complete(
        self,
        messages: Iterable[Message[Any]],
        functions: Iterable[Callable[..., FuncR]] | None = None,
        output_types: Iterable[type[R]] | None = None,
        *,
        stop: list[str] | None = None,
    ) -> (
        AssistantMessage[FunctionCall[FuncR]]
        | AssistantMessage[R]
        | AssistantMessage[str]
    ):
        """Request an LLM message."""
        if output_types is None:
            output_types = [] if functions else cast(list[type[R]], [str])

        function_schemas = [FunctionCallFunctionSchema(f) for f in functions or []] + [
            function_schema_for_type(type_)
            for type_ in output_types
            if not is_origin_subclass(type_, (str, StreamedStr))
        ]

        str_in_output_types = any(is_origin_subclass(cls, str) for cls in output_types)
        streamed_str_in_output_types = any(
            is_origin_subclass(cls, StreamedStr) for cls in output_types
        )
        allow_string_output = str_in_output_types or streamed_str_in_output_types

        openai_functions = [schema.dict() for schema in function_schemas]
        response = litellm_completion(
            model=self.model,
            messages=[message_to_openai_message(m) for m in messages],
            api_base=self.api_base,
            max_tokens=self.max_tokens,
            stop=stop,
            temperature=self.temperature,
            functions=openai_functions,
            function_call=(
                {"name": openai_functions[0]["name"]}
                if len(openai_functions) == 1 and not allow_string_output
                else None
            ),
        )

        first_chunk = next(response)
        response = chain([first_chunk], response)  # Replace first chunk

        function_schema = self._select_function_schema(first_chunk, function_schemas)
        if function_schema:
            try:
                content = function_schema.parse_args(
                    chunk.choices[0].delta.function_call.arguments
                    for chunk in response
                    if chunk.choices[0].delta.function_call
                    if chunk.choices[0].delta.function_call.arguments is not None
                )
                return AssistantMessage(content)  # type: ignore[return-value]
            except ValidationError as e:
                msg = (
                    "Failed to parse model output. You may need to update your prompt"
                    " to encourage the model to return a specific type."
                )
                raise StructuredOutputError(msg) from e

        if not allow_string_output:
            msg = (
                "String was returned by model but not expected. You may need to update"
                " your prompt to encourage the model to return a specific type."
            )
            raise ValueError(msg)
        streamed_str = StreamedStr(
            chunk.choices[0].delta.get("content", None)
            for chunk in response
            if chunk.choices[0].delta.get("content", None) is not None
        )
        if streamed_str_in_output_types:
            return AssistantMessage(streamed_str)  # type: ignore[return-value]
        return AssistantMessage(str(streamed_str))

    @overload
    async def acomplete(
        self,
        messages: Iterable[Message[Any]],
        functions: None = ...,
        output_types: None = ...,
        *,
        stop: list[str] | None = ...,
    ) -> AssistantMessage[str]:
        ...

    @overload
    async def acomplete(
        self,
        messages: Iterable[Message[Any]],
        functions: Iterable[Callable[..., FuncR]],
        output_types: None = ...,
        *,
        stop: list[str] | None = ...,
    ) -> AssistantMessage[FunctionCall[FuncR]] | AssistantMessage[str]:
        ...

    @overload
    async def acomplete(
        self,
        messages: Iterable[Message[Any]],
        functions: None = ...,
        output_types: Iterable[type[R]] = ...,
        *,
        stop: list[str] | None = ...,
    ) -> AssistantMessage[R]:
        ...

    @overload
    async def acomplete(
        self,
        messages: Iterable[Message[Any]],
        functions: Iterable[Callable[..., FuncR]],
        output_types: Iterable[type[R]],
        *,
        stop: list[str] | None = ...,
    ) -> AssistantMessage[FunctionCall[FuncR]] | AssistantMessage[R]:
        ...

    async def acomplete(
        self,
        messages: Iterable[Message[Any]],
        functions: Iterable[Callable[..., FuncR]] | None = None,
        output_types: Iterable[type[R]] | None = None,
        *,
        stop: list[str] | None = None,
    ) -> (
        AssistantMessage[FunctionCall[FuncR]]
        | AssistantMessage[R]
        | AssistantMessage[str]
    ):
        """Async version of `complete`."""
        if output_types is None:
            output_types = [] if functions else cast(list[type[R]], [str])

        function_schemas = [FunctionCallFunctionSchema(f) for f in functions or []] + [
            async_function_schema_for_type(type_)
            for type_ in output_types
            if not is_origin_subclass(type_, (str, AsyncStreamedStr))
        ]

        str_in_output_types = any(is_origin_subclass(cls, str) for cls in output_types)
        async_streamed_str_in_output_types = any(
            is_origin_subclass(cls, AsyncStreamedStr) for cls in output_types
        )
        allow_string_output = str_in_output_types or async_streamed_str_in_output_types

        openai_functions = [schema.dict() for schema in function_schemas]
        response = await litellm_acompletion(
            model=self.model,
            messages=[message_to_openai_message(m) for m in messages],
            api_base=self.api_base,
            max_tokens=self.max_tokens,
            stop=stop,
            temperature=self.temperature,
            functions=openai_functions,
            function_call=(
                {"name": openai_functions[0]["name"]}
                if len(openai_functions) == 1 and not allow_string_output
                else None
            ),
        )

        first_chunk = await anext(response)
        response = achain(async_iter([first_chunk]), response)  # Replace first chunk

        function_schema = self._select_function_schema(first_chunk, function_schemas)
        if function_schema:
            try:
                content = await function_schema.aparse_args(
                    chunk.choices[0].delta.function_call.arguments
                    async for chunk in response
                    if chunk.choices[0].delta.function_call
                    if chunk.choices[0].delta.function_call.arguments is not None
                )
                return AssistantMessage(content)  # type: ignore[return-value]
            except ValidationError as e:
                msg = (
                    "Failed to parse model output. You may need to update your prompt"
                    " to encourage the model to return a specific type."
                )
                raise StructuredOutputError(msg) from e

        if not allow_string_output:
            msg = (
                "String was returned by model but not expected. You may need to update"
                " your prompt to encourage the model to return a specific type."
            )
            raise ValueError(msg)
        async_streamed_str = AsyncStreamedStr(
            chunk.choices[0].delta.get("content", None)
            async for chunk in response
            if chunk.choices[0].delta.get("content", None) is not None
        )
        if async_streamed_str_in_output_types:
            return AssistantMessage(async_streamed_str)  # type: ignore[return-value]
        return AssistantMessage(await async_streamed_str.to_string())
