from collections.abc import AsyncIterator, Callable, Iterable, Iterator
from enum import Enum
from functools import singledispatch
from itertools import chain
from typing import Any, Literal, TypeVar, cast, overload

import openai
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam
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

    function_schema = (
        FunctionCallFunctionSchema(message.content.function)
        if isinstance(message.content, FunctionCall)
        else function_schema_for_type(type(message.content))
    )

    return {
        "role": OpenaiMessageRole.ASSISTANT.value,
        "content": None,
        "function_call": {
            "name": function_schema.name,
            "arguments": function_schema.serialize_args(message.content),
        },
    }


@message_to_openai_message.register(FunctionResultMessage)
def _(message: FunctionResultMessage[Any]) -> ChatCompletionMessageParam:
    function_schema = function_schema_for_type(type(message.content))
    return {
        "role": OpenaiMessageRole.FUNCTION.value,
        "name": FunctionCallFunctionSchema(message.function).name,
        "content": function_schema.serialize_args(message.content),
    }


def openai_chatcompletion_create(
    api_key: str | None,
    api_type: Literal["openai", "azure"],
    base_url: str | None,
    model: str,
    messages: list[ChatCompletionMessageParam],
    max_tokens: int | None = None,
    seed: int | None = None,
    stop: list[str] | None = None,
    temperature: float | None = None,
    functions: list[dict[str, Any]] | None = None,
    function_call: Literal["auto", "none"] | dict[str, Any] | None = None,
) -> Iterator[ChatCompletionChunk]:
    client_kwargs: dict[str, Any] = {
        "api_key": api_key,
    }
    if api_type == "openai" and base_url:
        client_kwargs["base_url"] = base_url

    client = (
        openai.AzureOpenAI(**client_kwargs)
        if api_type == "azure"
        else openai.OpenAI(**client_kwargs)
    )

    # `openai.OpenAI().chat.completions.create` doesn't accept `None` for some args
    # so only pass function args if there are functions
    kwargs: dict[str, Any] = {}
    if functions:
        kwargs["functions"] = functions
    if function_call:
        kwargs["function_call"] = function_call
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if stop is not None:
        kwargs["stop"] = stop

    response: Iterator[ChatCompletionChunk] = client.chat.completions.create(
        model=model,
        messages=messages,
        seed=seed,
        stream=True,
        temperature=temperature,
        **kwargs,
    )
    return response


async def openai_chatcompletion_acreate(
    api_key: str | None,
    api_type: Literal["openai", "azure"],
    base_url: str | None,
    model: str,
    messages: list[ChatCompletionMessageParam],
    max_tokens: int | None = None,
    seed: int | None = None,
    stop: list[str] | None = None,
    temperature: float | None = None,
    functions: list[dict[str, Any]] | None = None,
    function_call: Literal["auto", "none"] | dict[str, Any] | None = None,
) -> AsyncIterator[ChatCompletionChunk]:
    client_kwargs: dict[str, Any] = {
        "api_key": api_key,
    }
    if api_type == "openai" and base_url:
        client_kwargs["base_url"] = base_url

    client = (
        openai.AsyncAzureOpenAI(**client_kwargs)
        if api_type == "azure"
        else openai.AsyncOpenAI(**client_kwargs)
    )
    # `openai.AsyncOpenAI().chat.completions.create` doesn't accept `None` for some args
    # so only pass function args if there are functions
    kwargs: dict[str, Any] = {}
    if functions:
        kwargs["functions"] = functions
    if function_call:
        kwargs["function_call"] = function_call
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if stop is not None:
        kwargs["stop"] = stop

    response: AsyncIterator[ChatCompletionChunk] = await client.chat.completions.create(
        model=model,
        messages=messages,
        seed=seed,
        temperature=temperature,
        stream=True,
        **kwargs,
    )
    return response


BeseFunctionSchemaT = TypeVar("BeseFunctionSchemaT", bound=BaseFunctionSchema[Any])
R = TypeVar("R")
FuncR = TypeVar("FuncR")


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

    @staticmethod
    def _select_function_schema(
        chunk: ChatCompletionChunk,
        function_schemas: list[BeseFunctionSchemaT],
    ) -> BeseFunctionSchemaT | None:
        """Select the function schema based on the first response chunk."""
        if not chunk.choices[0].delta.function_call:
            return None

        function_name = chunk.choices[0].delta.function_call.name
        if function_name is None:
            msg = f"OpenAI function call name is None. Chunk: {chunk.model_dump_json()}"
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
        response = openai_chatcompletion_create(
            api_key=self.api_key,
            api_type=self.api_type,
            base_url=self.base_url,
            model=self.model,
            messages=[message_to_openai_message(m) for m in messages],
            max_tokens=self.max_tokens,
            seed=self.seed,
            stop=stop,
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

        function_schema = self._select_function_schema(first_chunk, function_schemas)
        if function_schema:
            try:
                content = function_schema.parse_args(
                    chunk.choices[0].delta.function_call.arguments
                    for chunk in response
                    if chunk.choices[0].delta.function_call
                    and chunk.choices[0].delta.function_call.arguments is not None
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
            chunk.choices[0].delta.content
            for chunk in response
            if chunk.choices[0].delta.content is not None
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
        response = await openai_chatcompletion_acreate(
            api_key=self.api_key,
            api_type=self.api_type,
            base_url=self.base_url,
            model=self.model,
            messages=[message_to_openai_message(m) for m in messages],
            max_tokens=self.max_tokens,
            seed=self.seed,
            stop=stop,
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

        function_schema = self._select_function_schema(first_chunk, function_schemas)
        if function_schema:
            try:
                content = await function_schema.aparse_args(
                    chunk.choices[0].delta.function_call.arguments
                    async for chunk in response
                    if chunk.choices[0].delta.function_call
                    and chunk.choices[0].delta.function_call.arguments is not None
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
            chunk.choices[0].delta.content
            async for chunk in response
            if chunk.choices[0].delta.content is not None
        )
        if async_streamed_str_in_output_types:
            return AssistantMessage(async_streamed_str)  # type: ignore[return-value]
        return AssistantMessage(await async_streamed_str.to_string())
