from collections.abc import AsyncIterator, Callable, Iterable, Iterator
from enum import Enum
from itertools import chain
from typing import Any, Literal, TypeVar, cast, overload

import openai
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam
from pydantic import ValidationError

from magentic.chat_model.base import ChatModel, StructuredOutputError
from magentic.chat_model.function_schema import (
    BaseFunctionSchema,
    FunctionCallFunctionSchema,
    function_schema_for_type,
)
from magentic.chat_model.message import (
    AssistantMessage,
    FunctionResultMessage,
    Message,
    SystemMessage,
    UserMessage,
)
from magentic.function_call import FunctionCall
from magentic.streaming import (
    AsyncStreamedStr,
    StreamedStr,
    achain,
    async_iter,
)
from magentic.typing import is_origin_subclass


class OpenaiMessageRole(Enum):
    ASSISTANT = "assistant"
    FUNCTION = "function"
    SYSTEM = "system"
    USER = "user"


def message_to_openai_message(message: Message[Any]) -> ChatCompletionMessageParam:
    """Convert a Message to an OpenAI message."""
    if isinstance(message, SystemMessage):
        return {
            "role": OpenaiMessageRole.SYSTEM.value,
            "content": message.content,
        }

    if isinstance(message, UserMessage):
        return {
            "role": OpenaiMessageRole.USER.value,
            "content": message.content,
        }

    if isinstance(message, AssistantMessage):
        if isinstance(message.content, str):
            return {
                "role": OpenaiMessageRole.ASSISTANT.value,
                "content": message.content,
            }

        function_schema: BaseFunctionSchema[Any]
        if isinstance(message.content, FunctionCall):
            function_schema = FunctionCallFunctionSchema(message.content.function)
        else:
            function_schema = function_schema_for_type(type(message.content))

        return {
            "role": OpenaiMessageRole.ASSISTANT.value,
            "content": None,
            "function_call": {
                "name": function_schema.name,
                "arguments": function_schema.serialize_args(message.content),
            },
        }

    if isinstance(message, FunctionResultMessage):
        function_schema = function_schema_for_type(type(message.content))
        return {
            "role": OpenaiMessageRole.FUNCTION.value,
            "name": function_schema.name,
            "content": function_schema.serialize_args(message.content),
        }

    raise NotImplementedError(type(message))


def openai_chatcompletion_create(
    api_type: Literal["openai", "azure"],
    model: str,
    messages: list[ChatCompletionMessageParam],
    max_tokens: int | None = None,
    temperature: float | None = None,
    functions: list[dict[str, Any]] | None = None,
    function_call: Literal["auto", "none"] | dict[str, Any] | None = None,
) -> Iterator[ChatCompletionChunk]:
    # `openai.OpenAI().chat.completions.create` doesn't accept `None` for some args
    # so only pass function args if there are functions
    kwargs: dict[str, Any] = {}
    if functions:
        kwargs["functions"] = functions
    if function_call:
        kwargs["function_call"] = function_call

    client = openai.AzureOpenAI() if api_type == "azure" else openai.OpenAI()
    response: Iterator[ChatCompletionChunk] = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
        **kwargs,
    )
    return response


async def openai_chatcompletion_acreate(
    api_type: Literal["openai", "azure"],
    model: str,
    messages: list[ChatCompletionMessageParam],
    max_tokens: int | None = None,
    temperature: float | None = None,
    functions: list[dict[str, Any]] | None = None,
    function_call: Literal["auto", "none"] | dict[str, Any] | None = None,
) -> AsyncIterator[ChatCompletionChunk]:
    # `openai.AsyncClient().chat.completions.create` doesn't accept `None` for some args
    # so only pass function args if there are functions
    kwargs: dict[str, Any] = {}
    if functions:
        kwargs["functions"] = functions
    if function_call:
        kwargs["function_call"] = function_call

    client = openai.AsyncAzureOpenAI() if api_type == "azure" else openai.AsyncClient()
    response: AsyncIterator[ChatCompletionChunk] = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
        **kwargs,
    )
    return response


R = TypeVar("R")
FuncR = TypeVar("FuncR")


class OpenaiChatModel(ChatModel):
    """An LLM chat model that uses the `openai` python package."""

    def __init__(
        self,
        model: str,
        *,
        api_type: Literal["openai", "azure"] = "openai",
        max_tokens: int | None = None,
        temperature: float | None = None,
    ):
        self._model = model
        self._api_type = api_type
        self._max_tokens = max_tokens
        self._temperature = temperature

    @property
    def model(self) -> str:
        return self._model

    @property
    def api_type(self) -> Literal["openai", "azure"]:
        return self._api_type

    @property
    def max_tokens(self) -> int | None:
        return self._max_tokens

    @property
    def temperature(self) -> float | None:
        return self._temperature

    @overload
    def complete(
        self,
        messages: Iterable[Message[Any]],
        functions: None = ...,
        output_types: None = ...,
    ) -> AssistantMessage[str]:
        ...

    @overload
    def complete(
        self,
        messages: Iterable[Message[Any]],
        functions: Iterable[Callable[..., FuncR]],
        output_types: None = ...,
    ) -> AssistantMessage[FunctionCall[FuncR]] | AssistantMessage[str]:
        ...

    @overload
    def complete(
        self,
        messages: Iterable[Message[Any]],
        functions: None = ...,
        output_types: Iterable[type[R]] = ...,
    ) -> AssistantMessage[R]:
        ...

    @overload
    def complete(
        self,
        messages: Iterable[Message[Any]],
        functions: Iterable[Callable[..., FuncR]],
        output_types: Iterable[type[R]],
    ) -> AssistantMessage[FunctionCall[FuncR]] | AssistantMessage[R]:
        ...

    def complete(
        self,
        messages: Iterable[Message[Any]],
        functions: Iterable[Callable[..., FuncR]] | None = None,
        output_types: Iterable[type[R | str]] | None = None,
    ) -> (
        AssistantMessage[FunctionCall[FuncR]]
        | AssistantMessage[R]
        | AssistantMessage[str]
    ):
        """Request an LLM message."""
        if output_types is None:
            output_types = [str]

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
        response = openai_chatcompletion_create(
            api_type=self.api_type,
            model=self.model,
            messages=[message_to_openai_message(m) for m in messages],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            functions=openai_functions,
            function_call=(
                {"name": openai_functions[0]["name"]}
                if len(openai_functions) == 1 and not allow_string_output
                else None
            ),
        )

        # Azure OpenAI sends a chunk with empty choices first
        first_chunk = next(response)
        if len(first_chunk.choices) == 0:
            first_chunk = next(response)

        response = chain([first_chunk], response)  # Replace first chunk
        first_chunk_delta = first_chunk.choices[0].delta

        if first_chunk_delta.function_call:
            function_schema_by_name = {
                function_schema.name: function_schema
                for function_schema in function_schemas
            }
            function_name = first_chunk_delta.function_call.name
            if function_name is None:
                msg = "OpenAI function call name is None"
                raise ValueError(msg)
            function_schema = function_schema_by_name[function_name]
            try:
                return AssistantMessage(
                    function_schema.parse_args(
                        chunk.choices[0].delta.function_call.arguments
                        for chunk in response
                        if chunk.choices[0].delta.function_call
                        and chunk.choices[0].delta.function_call.arguments is not None
                    )
                )
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
            chunk.choices[0].delta.content
            for chunk in response
            if chunk.choices[0].delta.content is not None
        )
        if streamed_str_in_output_types:
            return cast(AssistantMessage[R], AssistantMessage(streamed_str))
        return cast(AssistantMessage[R], AssistantMessage(str(streamed_str)))

    @overload
    async def acomplete(
        self,
        messages: Iterable[Message[Any]],
        functions: None = ...,
        output_types: None = ...,
    ) -> AssistantMessage[str]:
        ...

    @overload
    async def acomplete(
        self,
        messages: Iterable[Message[Any]],
        functions: Iterable[Callable[..., FuncR]],
        output_types: None = ...,
    ) -> AssistantMessage[FunctionCall[FuncR]] | AssistantMessage[str]:
        ...

    @overload
    async def acomplete(
        self,
        messages: Iterable[Message[Any]],
        functions: None = ...,
        output_types: Iterable[type[R]] = ...,
    ) -> AssistantMessage[R]:
        ...

    @overload
    async def acomplete(
        self,
        messages: Iterable[Message[Any]],
        functions: Iterable[Callable[..., FuncR]],
        output_types: Iterable[type[R]],
    ) -> AssistantMessage[FunctionCall[FuncR]] | AssistantMessage[R]:
        ...

    async def acomplete(
        self,
        messages: Iterable[Message[Any]],
        functions: Iterable[Callable[..., FuncR]] | None = None,
        output_types: Iterable[type[R | str]] | None = None,
    ) -> (
        AssistantMessage[FunctionCall[FuncR]]
        | AssistantMessage[R]
        | AssistantMessage[str]
    ):
        """Async version of `complete`."""
        if output_types is None:
            output_types = [str]

        function_schemas = [FunctionCallFunctionSchema(f) for f in functions or []] + [
            function_schema_for_type(type_)
            for type_ in output_types
            if not is_origin_subclass(type_, (str, AsyncStreamedStr))
        ]

        str_in_output_types = any(is_origin_subclass(cls, str) for cls in output_types)
        async_streamed_str_in_output_types = any(
            is_origin_subclass(cls, AsyncStreamedStr) for cls in output_types
        )
        allow_string_output = str_in_output_types or async_streamed_str_in_output_types

        openai_functions = [schema.dict() for schema in function_schemas]
        response = await openai_chatcompletion_acreate(
            api_type=self.api_type,
            model=self.model,
            messages=[message_to_openai_message(m) for m in messages],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            functions=openai_functions,
            function_call=(
                {"name": openai_functions[0]["name"]}
                if len(openai_functions) == 1 and not allow_string_output
                else None
            ),
        )

        # Azure OpenAI sends a chunk with empty choices first
        first_chunk = await anext(response)
        if len(first_chunk.choices) == 0:
            first_chunk = await anext(response)

        response = achain(async_iter([first_chunk]), response)  # Replace first chunk
        first_chunk_delta = first_chunk.choices[0].delta

        if first_chunk_delta.function_call:
            function_schema_by_name = {
                function_schema.name: function_schema
                for function_schema in function_schemas
            }
            function_name = first_chunk_delta.function_call.name
            if function_name is None:
                msg = "OpenAI function call name is None"
                raise ValueError(msg)
            function_schema = function_schema_by_name[function_name]
            try:
                return AssistantMessage(
                    await function_schema.aparse_args(
                        chunk.choices[0].delta.function_call.arguments
                        async for chunk in response
                        if chunk.choices[0].delta.function_call
                        and chunk.choices[0].delta.function_call.arguments is not None
                    )
                )
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
            chunk.choices[0].delta.content
            async for chunk in response
            if chunk.choices[0].delta.content is not None
        )
        if async_streamed_str_in_output_types:
            return cast(AssistantMessage[R], AssistantMessage(async_streamed_str))
        return cast(
            AssistantMessage[R], AssistantMessage(await async_streamed_str.to_string())
        )
