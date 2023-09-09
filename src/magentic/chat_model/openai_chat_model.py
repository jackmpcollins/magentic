import inspect
import json
import typing
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, AsyncIterator, Callable, Iterable, Iterator
from enum import Enum
from typing import Any, Generic, TypeVar, cast, get_args, get_origin

import openai
from pydantic import BaseModel, TypeAdapter, ValidationError, create_model

from magentic.chat_model.base import (
    AssistantMessage,
    FunctionCallMessage,
    FunctionResultMessage,
    Message,
    UserMessage,
)
from magentic.function_call import FunctionCall
from magentic.streaming import (
    AsyncStreamedStr,
    StreamedStr,
    aiter_streamed_json_array,
    iter_streamed_json_array,
)
from magentic.typing import is_origin_abstract, is_origin_subclass, name_type


class StructuredOutputError(Exception):
    ...


T = TypeVar("T")


class BaseFunctionSchema(ABC, Generic[T]):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    def description(self) -> str | None:
        return None

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        ...

    def dict(self) -> dict[str, Any]:
        schema = {"name": self.name, "parameters": self.parameters}
        if self.description:
            schema["description"] = self.description
        return schema

    @abstractmethod
    def parse_args(self, arguments: Iterable[str]) -> T:
        ...

    async def aparse_args(self, arguments: AsyncIterable[str]) -> T:
        # TODO: Convert AsyncIterable to lazy Iterable rather than list
        return self.parse_args([arg async for arg in arguments])

    def parse_args_to_message(self, arguments: Iterable[str]) -> AssistantMessage[T]:
        return AssistantMessage(self.parse_args(arguments))

    async def aparse_args_to_message(
        self, arguments: AsyncIterable[str]
    ) -> AssistantMessage[T]:
        return AssistantMessage(await self.aparse_args(arguments))

    @abstractmethod
    def serialize_args(self, value: T) -> str:
        ...


class Output(BaseModel, Generic[T]):
    value: T


class AnyFunctionSchema(BaseFunctionSchema[T], Generic[T]):
    def __init__(self, output_type: type[T]):
        self._output_type = output_type
        # https://github.com/python/mypy/issues/14458
        self._model = Output[output_type]  # type: ignore[valid-type]

    @property
    def name(self) -> str:
        return f"return_{name_type(self._output_type)}"

    @property
    def parameters(self) -> dict[str, Any]:
        model_schema = self._model.model_json_schema().copy()
        model_schema.pop("title", None)
        model_schema.pop("description", None)
        return model_schema

    def parse_args(self, arguments: Iterable[str]) -> T:
        return self._model.model_validate_json("".join(arguments)).value

    def serialize_args(self, value: T) -> str:
        return self._model(value=value).model_dump_json()


IterableT = TypeVar("IterableT", bound=Iterable[Any])


class IterableFunctionSchema(BaseFunctionSchema[IterableT], Generic[IterableT]):
    def __init__(self, output_type: type[IterableT]):
        self._output_type = output_type
        self._item_type_adapter = TypeAdapter(get_args(output_type)[0])
        # https://github.com/python/mypy/issues/14458
        self._model = Output[output_type]  # type: ignore[valid-type]

    @property
    def name(self) -> str:
        return f"return_{name_type(self._output_type)}"

    @property
    def parameters(self) -> dict[str, Any]:
        model_schema = self._model.model_json_schema().copy()
        model_schema.pop("title", None)
        model_schema.pop("description", None)
        return model_schema

    def parse_args(self, arguments: Iterable[str]) -> IterableT:
        iter_items = (
            self._item_type_adapter.validate_json(item)
            for item in iter_streamed_json_array(arguments)
        )
        return self._model.model_validate({"value": iter_items}).value

    def serialize_args(self, value: IterableT) -> str:
        return self._model(value=value).model_dump_json()


AsyncIterableT = TypeVar("AsyncIterableT", bound=AsyncIterable[Any])


class AsyncIterableFunctionSchema(
    BaseFunctionSchema[AsyncIterableT], Generic[AsyncIterableT]
):
    def __init__(self, output_type: type[AsyncIterableT]):
        self._output_type = output_type
        self._item_type_adapter = TypeAdapter(get_args(output_type)[0])
        # Convert to list so pydantic can handle for schema generation
        # But keep the type hint using AsyncIterableT for type checking
        self._model: type[Output[AsyncIterableT]] = Output[list[get_args(output_type)[0]]]  # type: ignore

    @property
    def name(self) -> str:
        return f"return_{name_type(self._output_type)}"

    @property
    def parameters(self) -> dict[str, Any]:
        model_schema = self._model.model_json_schema().copy()
        model_schema.pop("title", None)
        model_schema.pop("description", None)
        return model_schema

    def parse_args(self, arguments: Iterable[str]) -> AsyncIterableT:
        raise NotImplementedError()

    async def aparse_args(self, arguments: AsyncIterable[str]) -> AsyncIterableT:
        aiter_items = (
            self._item_type_adapter.validate_json(item)
            async for item in aiter_streamed_json_array(arguments)
        )
        if (get_origin(self._output_type) or self._output_type) in (
            typing.AsyncIterable,
            typing.AsyncIterator,
        ) or is_origin_abstract(self._output_type):
            return cast(AsyncIterableT, aiter_items)

        raise NotImplementedError()

    def serialize_args(self, value: AsyncIterableT) -> str:
        return self._model(value=value).model_dump_json()


class DictFunctionSchema(BaseFunctionSchema[T], Generic[T]):
    def __init__(self, output_type: type[T]):
        self._output_type = output_type
        self._type_adapter: TypeAdapter[T] = TypeAdapter(output_type)

    @property
    def name(self) -> str:
        return f"return_{name_type(self._output_type)}"

    @property
    def parameters(self) -> dict[str, Any]:
        model_schema = self._type_adapter.json_schema().copy()
        model_schema["properties"] = model_schema.get("properties", {})
        return model_schema

    def parse_args(self, arguments: Iterable[str]) -> T:
        return self._type_adapter.validate_json("".join(arguments))

    def serialize_args(self, value: T) -> str:
        return self._type_adapter.dump_json(value).decode()


BaseModelT = TypeVar("BaseModelT", bound=BaseModel)


class BaseModelFunctionSchema(BaseFunctionSchema[BaseModelT], Generic[BaseModelT]):
    def __init__(self, model: type[BaseModelT]):
        self._model = model

    @property
    def name(self) -> str:
        return f"return_{self._model.__name__.lower()}"

    @property
    def parameters(self) -> dict[str, Any]:
        model_schema = self._model.model_json_schema().copy()
        model_schema.pop("title", None)
        model_schema.pop("description", None)
        return model_schema

    def parse_args(self, arguments: Iterable[str]) -> BaseModelT:
        return self._model.model_validate_json("".join(arguments))

    def serialize_args(self, value: BaseModelT) -> str:
        return value.model_dump_json()


class FunctionCallFunctionSchema(BaseFunctionSchema[FunctionCall[T]], Generic[T]):
    def __init__(self, func: Callable[..., T]):
        self._func = func
        # https://github.com/pydantic/pydantic/issues/3585#issuecomment-1002745763
        fields: dict[str, Any] = {
            param.name: (
                (param.annotation if param.annotation != inspect._empty else Any),
                (param.default if param.default != inspect._empty else ...),
            )
            for param in inspect.signature(func).parameters.values()
        }
        self._model = create_model("FuncModel", **fields)

    @property
    def name(self) -> str:
        return self._func.__name__

    @property
    def description(self) -> str | None:
        return inspect.getdoc(self._func)

    @property
    def parameters(self) -> dict[str, Any]:
        schema: dict[str, Any] = self._model.model_json_schema().copy()
        schema.pop("title", None)
        return schema

    def parse_args(self, arguments: Iterable[str]) -> FunctionCall[T]:
        args = self._model.model_validate_json("".join(arguments)).model_dump(
            exclude_unset=True
        )
        return FunctionCall(self._func, **args)

    async def aparse_args(self, arguments: AsyncIterable[str]) -> FunctionCall[T]:
        return self.parse_args([arg async for arg in arguments])

    def parse_args_to_message(self, arguments: Iterable[str]) -> FunctionCallMessage[T]:
        return FunctionCallMessage(self.parse_args(arguments))

    async def aparse_args_to_message(
        self, arguments: AsyncIterable[str]
    ) -> FunctionCallMessage[T]:
        return FunctionCallMessage(await self.aparse_args(arguments))

    def serialize_args(self, value: FunctionCall[T]) -> str:
        return json.dumps(value.arguments)


# TODO: Add type hints here. Possibly use `functools.singledispatch` instead.
def function_schema_for_type(type_: type[Any]) -> BaseFunctionSchema[Any]:
    """Create a FunctionSchema for the given type."""
    if is_origin_subclass(type_, BaseModel):
        return BaseModelFunctionSchema(type_)

    if is_origin_subclass(type_, dict):
        return DictFunctionSchema(type_)

    if is_origin_subclass(type_, Iterable):
        return IterableFunctionSchema(type_)

    if is_origin_subclass(type_, AsyncIterable):
        return AsyncIterableFunctionSchema(type_)

    return AnyFunctionSchema(type_)


class OpenaiMessageRole(Enum):
    ASSISTANT = "assistant"
    FUNCTION = "function"
    SYSTEM = "system"
    USER = "user"


def message_to_openai_message(message: Message[Any]) -> dict[str, Any]:
    """Convert a `Message` to an OpenAI message dict."""
    if isinstance(message, UserMessage):
        return {"role": OpenaiMessageRole.USER.value, "content": message.content}

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
        return {
            "role": OpenaiMessageRole.FUNCTION.value,
            "name": FunctionCallFunctionSchema(message.function_call.function).name,
            "content": json.dumps(message.content),
        }

    raise NotImplementedError(type(message))


R = TypeVar("R")
FuncR = TypeVar("FuncR")


class OpenaiChatModel:
    def __init__(self, model: str = "gpt-3.5-turbo-0613", temperature: float = 0):
        self._model = model
        self._temperature = temperature

    def complete(
        self,
        messages: Iterable[Message[Any]],
        functions: Iterable[Callable[..., FuncR]] | None = None,
        output_types: Iterable[type[R | str]] | None = None,
    ) -> FunctionCallMessage[FuncR] | AssistantMessage[R]:
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

        # `openai.ChatCompletion.create` doesn't accept `None`
        # so only pass function args if there are functions
        function_args: dict[str, Any] = {}
        if function_schemas:
            function_args["functions"] = [schema.dict() for schema in function_schemas]
        if len(function_schemas) == 1 and not allow_string_output:
            # Force the model to call the function
            function_args["function_call"] = {"name": function_schemas[0].name}

        response: Iterator[dict[str, Any]] = openai.ChatCompletion.create(  # type: ignore[no-untyped-call]
            model=self._model,
            messages=[message_to_openai_message(m) for m in messages],
            temperature=self._temperature,
            **function_args,
            stream=True,
        )

        first_chunk = next(response)
        first_chunk_delta = first_chunk["choices"][0]["delta"]

        if first_chunk_delta.get("function_call"):
            function_schema_by_name = {
                function_schema.name: function_schema
                for function_schema in function_schemas
            }
            function_name = first_chunk_delta["function_call"]["name"]
            function_schema = function_schema_by_name[function_name]
            try:
                message = function_schema.parse_args_to_message(
                    chunk["choices"][0]["delta"]["function_call"]["arguments"]
                    for chunk in response
                    if chunk["choices"][0]["delta"]
                )
            except ValidationError as e:
                raise StructuredOutputError(
                    "Failed to parse model output. You may need to update your prompt"
                    " to encourage the model to return a specific type."
                ) from e
            return message

        if not allow_string_output:
            raise ValueError(
                "String was returned by model but not expected. You may need to update"
                " your prompt to encourage the model to return a specific type."
            )
        streamed_str = StreamedStr(
            chunk["choices"][0]["delta"]["content"]
            for chunk in response
            if chunk["choices"][0]["delta"]
        )
        if streamed_str_in_output_types:
            return cast(AssistantMessage[R], AssistantMessage(streamed_str))
        return cast(AssistantMessage[R], AssistantMessage(str(streamed_str)))

    # TODO: Deduplicate this and `complete`
    async def acomplete(
        self,
        messages: Iterable[Message[Any]],
        functions: Iterable[Callable[..., FuncR]] | None = None,
        output_types: Iterable[type[R | str]] | None = None,
    ) -> FunctionCallMessage[FuncR] | AssistantMessage[R]:
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

        # `openai.ChatCompletion.acreate` doesn't accept `None`
        # so only pass function args if there are functions
        function_args: dict[str, Any] = {}
        if function_schemas:
            function_args["functions"] = [schema.dict() for schema in function_schemas]
        if len(function_schemas) == 1 and not allow_string_output:
            # Force the model to call the function
            function_args["function_call"] = {"name": function_schemas[0].name}

        response: AsyncIterator[dict[str, Any]] = await openai.ChatCompletion.acreate(  # type: ignore[no-untyped-call]
            model=self._model,
            messages=[message_to_openai_message(m) for m in messages],
            temperature=self._temperature,
            **function_args,
            stream=True,
        )

        first_chunk = await anext(response)
        first_chunk_delta = first_chunk["choices"][0]["delta"]

        if first_chunk_delta.get("function_call"):
            function_schema_by_name = {
                function_schema.name: function_schema
                for function_schema in function_schemas
            }
            function_name = first_chunk_delta["function_call"]["name"]
            function_schema = function_schema_by_name[function_name]
            try:
                message = await function_schema.aparse_args_to_message(
                    chunk["choices"][0]["delta"]["function_call"]["arguments"]
                    async for chunk in response
                    if chunk["choices"][0]["delta"]
                )
            except ValidationError as e:
                raise StructuredOutputError(
                    "Failed to parse model output. You may need to update your prompt"
                    " to encourage the model to return a specific type."
                ) from e
            return message

        if not allow_string_output:
            raise ValueError(
                "String was returned by model but not expected. You may need to update"
                " your prompt to encourage the model to return a specific type."
            )
        async_streamed_str = AsyncStreamedStr(
            chunk["choices"][0]["delta"]["content"]
            async for chunk in response
            if chunk["choices"][0]["delta"]
        )
        if async_streamed_str_in_output_types:
            return cast(AssistantMessage[R], AssistantMessage(async_streamed_str))
        return cast(
            AssistantMessage[R], AssistantMessage(await async_streamed_str.to_string())
        )
