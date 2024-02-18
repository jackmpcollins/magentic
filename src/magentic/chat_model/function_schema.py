import inspect
import typing
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, Callable, Iterable
from typing import Any, Generic, TypeVar, cast, get_args, get_origin

from pydantic import BaseModel, TypeAdapter, create_model

from magentic.function_call import FunctionCall
from magentic.streaming import (
    aiter_streamed_json_array,
    iter_streamed_json_array,
)
from magentic.typing import is_origin_abstract, is_origin_subclass, name_type

T = TypeVar("T")


class BaseFunctionSchema(ABC, Generic[T]):
    """Converts a Python object to the JSON Schema that represents it as a function for the LLM."""

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
        return self.parse_args([arg async for arg in arguments])

    @abstractmethod
    def serialize_args(self, value: T) -> str:
        ...


class Output(BaseModel, Generic[T]):
    value: T


class AnyFunctionSchema(BaseFunctionSchema[T], Generic[T]):
    """The most generic FunctionSchema that should work for most types supported by pydantic."""

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
    """FunctionSchema for types that are iterable. Can parse LLM output as a stream."""

    def __init__(self, output_type: type[IterableT]):
        self._output_type = output_type
        self._item_type_adapter = TypeAdapter(
            args[0] if (args := get_args(output_type)) else Any
        )
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
    """FunctionSchema for types that are async iterable. Can parse LLM output as a stream."""

    def __init__(self, output_type: type[AsyncIterableT]):
        self._output_type = output_type
        item_type = args[0] if (args := get_args(output_type)) else Any
        self._item_type_adapter = TypeAdapter(item_type)
        # Convert to list so pydantic can handle for schema generation
        # But keep the type hint using AsyncIterableT for type checking
        self._model = Output[list[item_type]]  # type: ignore[valid-type]

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
        raise NotImplementedError

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

        raise NotImplementedError

    def serialize_args(self, value: AsyncIterableT) -> str:
        raise NotImplementedError


class DictFunctionSchema(BaseFunctionSchema[T], Generic[T]):
    """FunctionSchema for dict."""

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
    """FunctionSchema for pydantic BaseModel."""

    def __init__(self, model: type[BaseModelT]):
        self._model = model

    @property
    def name(self) -> str:
        return f"return_{name_type(self._model)}"

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


def create_model_from_function(func: Callable[..., Any]) -> type[BaseModel]:
    """Create a Pydantic model from a function signature."""
    # https://github.com/pydantic/pydantic/issues/3585#issuecomment-1002745763
    fields: dict[str, Any] = {}
    for param in inspect.signature(func).parameters.values():
        # *args
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            fields[param.name] = (
                (
                    list[param.annotation]  # type: ignore[name-defined]
                    if param.annotation != inspect._empty
                    else list[Any]
                ),
                param.default if param.default != inspect._empty else [],
            )
            continue

        # **kwargs
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            fields[param.name] = (
                dict[str, param.annotation]  # type: ignore[name-defined]
                if param.annotation != inspect._empty
                else dict[str, Any],
                param.default if param.default != inspect._empty else {},
            )
            continue

        fields[param.name] = (
            (param.annotation if param.annotation != inspect._empty else Any),
            (param.default if param.default != inspect._empty else ...),
        )
    return create_model("FuncModel", **fields)


class FunctionCallFunctionSchema(BaseFunctionSchema[FunctionCall[T]], Generic[T]):
    """FunctionSchema for FunctionCall."""

    def __init__(self, func: Callable[..., T]):
        self._func = func
        self._model = create_model_from_function(func)

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
        model = self._model.model_validate_json("".join(arguments))
        supplied_params = [
            param
            for param in inspect.signature(self._func).parameters.values()
            if param.name in model.model_fields_set
        ]

        args_positional_only = [
            getattr(model, param.name)
            for param in supplied_params
            if param.kind == param.POSITIONAL_ONLY
        ]
        args_positional_or_keyword = [
            getattr(model, param.name)
            for param in supplied_params
            if param.kind == param.POSITIONAL_OR_KEYWORD
        ]
        args_var_positional = [
            arg
            for param in supplied_params
            if param.kind == param.VAR_POSITIONAL
            for arg in getattr(model, param.name)
        ]
        args_keyword_only = {
            param.name: getattr(model, param.name)
            for param in supplied_params
            if param.kind == param.KEYWORD_ONLY
        }
        args_var_keyword = {
            name: value
            for param in supplied_params
            if param.kind == param.VAR_KEYWORD
            for name, value in getattr(model, param.name).items()
        }
        return FunctionCall(
            self._func,
            *args_positional_only,
            *args_positional_or_keyword,
            *args_var_positional,
            **args_keyword_only,
            **args_var_keyword,
        )

    def serialize_args(self, value: FunctionCall[T]) -> str:
        return self._model(**value.arguments).model_dump_json(exclude_unset=True)


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
