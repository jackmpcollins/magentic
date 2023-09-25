from collections.abc import AsyncIterator, Callable, Iterable, Iterator
from enum import Enum
from typing import Any, Literal, TypeVar, cast

import openai
from pydantic import BaseModel, ValidationError

from magentic.chat_model.function_schema import (
    BaseFunctionSchema,
    FunctionCallFunctionSchema,
    function_schema_for_type,
)
from magentic.chat_model.message import (
    AssistantMessage,
    FunctionResultMessage,
    Message,
    UserMessage,
)
from magentic.function_call import FunctionCall
from magentic.settings import get_settings
from magentic.streaming import (
    AsyncStreamedStr,
    StreamedStr,
)
from magentic.typing import is_origin_subclass


class StructuredOutputError(Exception):
    """Raised when the LLM output could not be parsed."""


class OpenaiMessageRole(Enum):
    ASSISTANT = "assistant"
    FUNCTION = "function"
    SYSTEM = "system"
    USER = "user"


class OpenaiChatCompletionFunctionCall(BaseModel):
    name: str | None = None
    arguments: str

    def get_name_or_raise(self) -> str:
        """Return the name, raising an error if it doesn't exist."""
        if self.name is None:
            msg = "OpenAI function call name is None"
            raise ValueError(msg)
        return self.name


class OpenaiChatCompletionDelta(BaseModel):
    role: OpenaiMessageRole | None = None
    content: str | None = None
    function_call: OpenaiChatCompletionFunctionCall | None = None


class OpenaiChatCompletionChunkChoice(BaseModel):
    delta: OpenaiChatCompletionDelta


class OpenaiChatCompletionChunk(BaseModel):
    choices: list[OpenaiChatCompletionChunkChoice]


class OpenaiChatCompletionChoiceMessage(BaseModel):
    role: OpenaiMessageRole
    name: str | None = None
    content: str | None
    function_call: OpenaiChatCompletionFunctionCall | None = None


class OpenaiChatCompletionChoice(BaseModel):
    message: OpenaiChatCompletionDelta


class OpenaiChatCompletion(BaseModel):
    choices: list[OpenaiChatCompletionChoice]


def message_to_openai_message(
    message: Message[Any],
) -> OpenaiChatCompletionChoiceMessage:
    """Convert a Message to an OpenAI message."""
    if isinstance(message, UserMessage):
        return OpenaiChatCompletionChoiceMessage(
            role=OpenaiMessageRole.USER, content=message.content
        )

    if isinstance(message, AssistantMessage):
        if isinstance(message.content, str):
            return OpenaiChatCompletionChoiceMessage(
                role=OpenaiMessageRole.ASSISTANT, content=message.content
            )

        function_schema: BaseFunctionSchema[Any]
        if isinstance(message.content, FunctionCall):
            function_schema = FunctionCallFunctionSchema(message.content.function)
        else:
            function_schema = function_schema_for_type(type(message.content))

        return OpenaiChatCompletionChoiceMessage(
            role=OpenaiMessageRole.ASSISTANT,
            content=None,
            function_call=OpenaiChatCompletionFunctionCall(
                name=function_schema.name,
                arguments=function_schema.serialize_args(message.content),
            ),
        )

    if isinstance(message, FunctionResultMessage):
        function_schema = function_schema_for_type(type(message.content))
        return OpenaiChatCompletionChoiceMessage(
            role=OpenaiMessageRole.FUNCTION,
            name=function_schema.name,
            content=function_schema.serialize_args(message.content),
        )

    raise NotImplementedError(type(message))


def openai_chatcompletion_create(
    model: str,
    messages: Iterable[OpenaiChatCompletionChoiceMessage],
    temperature: float | None = None,
    functions: list[dict[str, Any]] | None = None,
    function_call: Literal["auto", "none"] | dict[str, Any] | None = None,
) -> Iterator[OpenaiChatCompletionChunk]:
    """Type-annotated version of `openai.ChatCompletion.create`."""
    # `openai.ChatCompletion.create` doesn't accept `None`
    # so only pass function args if there are functions
    kwargs: dict[str, Any] = {}
    if functions:
        kwargs["functions"] = functions
    if function_call:
        kwargs["function_call"] = function_call

    response: Iterator[dict[str, Any]] = openai.ChatCompletion.create(  # type: ignore[no-untyped-call]
        model=model,
        messages=[m.model_dump(mode="json", exclude_unset=True) for m in messages],
        temperature=temperature,
        stream=True,
        **kwargs,
    )
    return (OpenaiChatCompletionChunk.model_validate(chunk) for chunk in response)


async def openai_chatcompletion_acreate(
    model: str,
    messages: Iterable[OpenaiChatCompletionChoiceMessage],
    temperature: float | None = None,
    functions: list[dict[str, Any]] | None = None,
    function_call: Literal["auto", "none"] | dict[str, Any] | None = None,
) -> AsyncIterator[OpenaiChatCompletionChunk]:
    """Type-annotated version of `openai.ChatCompletion.acreate`."""
    # `openai.ChatCompletion.create` doesn't accept `None`
    # so only pass function args if there are functions
    kwargs: dict[str, Any] = {}
    if functions:
        kwargs["functions"] = functions
    if function_call:
        kwargs["function_call"] = function_call

    response: AsyncIterator[dict[str, Any]] = await openai.ChatCompletion.acreate(  # type: ignore[no-untyped-call]
        model=model,
        messages=[m.model_dump(mode="json", exclude_unset=True) for m in messages],
        temperature=temperature,
        stream=True,
        **kwargs,
    )
    return (OpenaiChatCompletionChunk.model_validate(chunk) async for chunk in response)


R = TypeVar("R")
FuncR = TypeVar("FuncR")


class OpenaiChatModel:
    """An LLM chat model that uses the `openai` python package."""

    def __init__(self, model: str | None = None, temperature: float | None = None):
        self._model = model
        self._temperature = temperature

    @property
    def model(self) -> str:
        if self._model is not None:
            return self._model
        return get_settings().openai_model

    @property
    def temperature(self) -> float | None:
        if self._temperature is not None:
            return self._temperature
        return get_settings().openai_temperature

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
            model=self.model,
            messages=[message_to_openai_message(m) for m in messages],
            temperature=self.temperature,
            functions=openai_functions,
            function_call=(
                {"name": openai_functions[0]["name"]}
                if len(openai_functions) == 1 and not allow_string_output
                else None
            ),
        )

        first_chunk = next(response)
        first_chunk_delta = first_chunk.choices[0].delta

        if first_chunk_delta.function_call:
            function_schema_by_name = {
                function_schema.name: function_schema
                for function_schema in function_schemas
            }
            function_name = first_chunk_delta.function_call.get_name_or_raise()
            function_schema = function_schema_by_name[function_name]
            try:
                return AssistantMessage(
                    function_schema.parse_args(
                        chunk.choices[0].delta.function_call.arguments
                        for chunk in response
                        if chunk.choices[0].delta.function_call
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
            model=self.model,
            messages=[message_to_openai_message(m) for m in messages],
            temperature=self.temperature,
            functions=openai_functions,
            function_call=(
                {"name": openai_functions[0]["name"]}
                if len(openai_functions) == 1 and not allow_string_output
                else None
            ),
        )

        first_chunk = await anext(response)
        first_chunk_delta = first_chunk.choices[0].delta

        if first_chunk_delta.function_call:
            function_schema_by_name = {
                function_schema.name: function_schema
                for function_schema in function_schemas
            }
            function_name = first_chunk_delta.function_call.get_name_or_raise()
            function_schema = function_schema_by_name[function_name]
            try:
                return AssistantMessage(
                    await function_schema.aparse_args(
                        chunk.choices[0].delta.function_call.arguments
                        async for chunk in response
                        if chunk.choices[0].delta.function_call
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
