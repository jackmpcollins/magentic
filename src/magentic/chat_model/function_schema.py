import inspect
import typing
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, Callable, Iterable
from functools import singledispatch
from typing import Any, Generic, TypeVar, cast, get_args, get_origin

from pydantic import BaseModel, TypeAdapter, create_model

from magentic.function_call import FunctionCall
from magentic.streaming import (
    aiter_streamed_json_array,
    iter_streamed_json_array,
)
from magentic.typing import is_origin_abstract, name_type

T = TypeVar("T")


class BaseFunctionSchema(ABC, Generic[T]):
    """Converts a Python object to the JSON Schema that represents it as a function for the LLM."""

    # Allow any arguments to avoid error passing type to subclasses without __init__
    def __init__(self, *args: Any, **kwargs: Any):
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the function.

        Must be a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum length of 64.
        """
        ...

    @property
    def description(self) -> str | None:
        """A description of what the function does."""
        return None

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """The parameters the functions accepts as a JSON Schema object."""
        ...

    def dict(self) -> dict[str, Any]:
        schema = {"name": self.name, "parameters": self.parameters}
        if self.description:
            schema["description"] = self.description
        return schema


class AsyncFunctionSchema(BaseFunctionSchema[T], Generic[T]):
    @abstractmethod
    async def aparse_args(self, chunks: AsyncIterable[str]) -> T:
        """Parse an async iterable of string chunks into the function arguments."""
        ...

    @abstractmethod
    async def aserialize_args(self, value: T) -> str:
        """Serialize the function arguments into a JSON string."""
        ...


class FunctionSchema(AsyncFunctionSchema[T], Generic[T]):
    @abstractmethod
    def parse_args(self, chunks: Iterable[str]) -> T:
        """Parse an iterable of string chunks into the function arguments."""
        ...

    @abstractmethod
    def serialize_args(self, value: T) -> str:
        """Serialize the function arguments into a JSON string."""
        ...

    async def aparse_args(self, chunks: AsyncIterable[str]) -> T:
        """Parse an async iterable of string chunks into the function arguments."""
        return self.parse_args([chunk async for chunk in chunks])

    async def aserialize_args(self, value: T) -> str:
        """Serialize the function arguments into a JSON string."""
        return self.serialize_args(value)


# Use the singledispatch registry to map classes to FunctionSchemas
# because this handles subclass resolution for us.
@singledispatch
def _async_function_schema_registry(type_: type[T]) -> AsyncFunctionSchema[T]:
    msg = f"No FunctionSchema registered for type {type_}"
    raise TypeError(msg)


def async_function_schema_for_type(type_: type[T]) -> AsyncFunctionSchema[T]:
    """Create a FunctionSchema for the given type."""
    function_schema_cls = _async_function_schema_registry.dispatch(
        get_origin(type_) or type_
    )
    return function_schema_cls(type_)


@singledispatch
def _function_schema_registry(type_: type[T]) -> FunctionSchema[T]:
    msg = f"No FunctionSchema registered for type {type_}"
    raise TypeError(msg)


def function_schema_for_type(type_: type[T]) -> FunctionSchema[T]:
    """Create a FunctionSchema for the given type."""
    function_schema_cls = _function_schema_registry.dispatch(get_origin(type_) or type_)
    return function_schema_cls(type_)


TypeFunctionSchemaT = TypeVar(
    "TypeFunctionSchemaT", bound=type[BaseFunctionSchema[Any]]
)


def register_function_schema(
    type_: type[Any],
) -> Callable[[TypeFunctionSchemaT], TypeFunctionSchemaT]:
    """Register a new FunctionSchema for the given type."""

    def _register(cls: TypeFunctionSchemaT) -> TypeFunctionSchemaT:
        if cls.__abstractmethods__:
            msg = f"FunctionSchema {cls} has not implemented abstract methods: {cls.__abstractmethods__}"
            raise TypeError(msg)

        if issubclass(cls, AsyncFunctionSchema):
            _async_function_schema_registry.register(type_, cls)
        if issubclass(cls, FunctionSchema):
            _function_schema_registry.register(type_, cls)
        return cls  # type: ignore[return-value]

    return _register


class Output(BaseModel, Generic[T]):
    value: T


@register_function_schema(object)
class AnyFunctionSchema(FunctionSchema[T], Generic[T]):
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

    def parse_args(self, chunks: Iterable[str]) -> T:
        args_json = "".join(chunks)
        return self._model.model_validate_json(args_json).value

    def serialize_args(self, value: T) -> str:
        return self._model(value=value).model_dump_json()


IterableT = TypeVar("IterableT", bound=Iterable[Any])


@register_function_schema(Iterable)
class IterableFunctionSchema(FunctionSchema[IterableT], Generic[IterableT]):
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

    def parse_args(self, chunks: Iterable[str]) -> IterableT:
        iter_items = (
            self._item_type_adapter.validate_json(item)
            for item in iter_streamed_json_array(chunks)
        )
        return self._model.model_validate({"value": iter_items}).value

    def serialize_args(self, value: IterableT) -> str:
        return self._model(value=value).model_dump_json()


AsyncIterableT = TypeVar("AsyncIterableT", bound=AsyncIterable[Any])


@register_function_schema(AsyncIterable)
class AsyncIterableFunctionSchema(
    AsyncFunctionSchema[AsyncIterableT], Generic[AsyncIterableT]
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

    async def aparse_args(self, chunks: AsyncIterable[str]) -> AsyncIterableT:
        aiter_items = (
            self._item_type_adapter.validate_json(item)
            async for item in aiter_streamed_json_array(chunks)
        )
        if (get_origin(self._output_type) or self._output_type) in (
            typing.AsyncIterable,
            typing.AsyncIterator,
        ) or is_origin_abstract(self._output_type):
            return cast(AsyncIterableT, aiter_items)

        raise NotImplementedError

    async def aserialize_args(self, value: AsyncIterableT) -> str:
        return self._model(value=[chunk async for chunk in value]).model_dump_json()


@register_function_schema(dict)
class DictFunctionSchema(FunctionSchema[T], Generic[T]):
    """FunctionSchema for dict."""

    def __init__(self, output_type: type[T]):
        self._output_type = output_type
        self._type_adapter = TypeAdapter(output_type)

    @property
    def name(self) -> str:
        return f"return_{name_type(self._output_type)}"

    @property
    def parameters(self) -> dict[str, Any]:
        model_schema = self._type_adapter.json_schema().copy()
        model_schema["properties"] = model_schema.get("properties", {})
        return model_schema

    def parse_args(self, chunks: Iterable[str]) -> T:
        args_json = "".join(chunks)
        return self._type_adapter.validate_json(args_json)

    def serialize_args(self, value: T) -> str:
        return self._type_adapter.dump_json(value).decode()


BaseModelT = TypeVar("BaseModelT", bound=BaseModel)


@register_function_schema(BaseModel)
class BaseModelFunctionSchema(FunctionSchema[BaseModelT], Generic[BaseModelT]):
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

    def parse_args(self, chunks: Iterable[str]) -> BaseModelT:
        args_json = "".join(chunks)
        return self._model.model_validate_json(args_json)

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


class FunctionCallFunctionSchema(FunctionSchema[FunctionCall[T]], Generic[T]):
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

    def parse_args(self, chunks: Iterable[str]) -> FunctionCall[T]:
        args_json = "".join(chunks)
        model = self._model.model_validate_json(args_json)
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
