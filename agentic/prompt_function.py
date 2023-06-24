import inspect
import types
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import update_wrapper
from typing import (
    Any,
    Callable,
    Generic,
    ParamSpec,
    Sequence,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

import openai
from pydantic import BaseModel, validate_arguments
from pydantic.generics import GenericModel

from agentic.function_call import FunctionCall

T = TypeVar("T")


class Output(GenericModel, Generic[T]):
    value: T


class BaseFunctionSchema(ABC, Generic[T]):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    def description(self) -> str | None:
        return None

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        pass

    def dict(self) -> dict[str, Any]:
        schema = {"name": self.name, "parameters": self.parameters}
        if self.description:
            schema["description"] = self.description
        return schema

    @abstractmethod
    def parse(self, arguments: str) -> T:
        pass


class AnyFunctionSchema(BaseFunctionSchema[T], Generic[T]):
    def __init__(self, return_type: type[T]):
        self._return_type = return_type
        # https://github.com/python/mypy/issues/14458
        self._model = Output[return_type]  # type: ignore[valid-type]

    @property
    def name(self) -> str:
        return f"return_{self._return_type.__name__.lower()}"

    @property
    def parameters(self) -> dict[str, Any]:
        model_schema = self._model.schema().copy()
        model_schema.pop("title", None)
        model_schema.pop("description", None)
        return model_schema

    def parse(self, arguments: str) -> T:
        return self._model.parse_raw(arguments).value


BaseModelT = TypeVar("BaseModelT", bound=BaseModel)


class BaseModelFunctionSchema(BaseFunctionSchema[BaseModelT], Generic[BaseModelT]):
    def __init__(self, model: type[BaseModelT]):
        self._model = model

    @property
    def name(self) -> str:
        return f"return_{self._model.__name__.lower()}"

    @property
    def parameters(self) -> dict[str, Any]:
        model_schema = self._model.schema().copy()
        model_schema.pop("title", None)
        model_schema.pop("description", None)
        return model_schema

    def parse(self, arguments: str) -> BaseModelT:
        return self._model.parse_raw(arguments)


class FunctionCallFunctionSchema(BaseFunctionSchema[FunctionCall[T]], Generic[T]):
    def __init__(self, func: Callable[..., T]):
        self._func = func
        # https://github.com/python/mypy/issues/2087
        self._model: BaseModel = validate_arguments(self._func).model  # type: ignore[attr-defined]
        self._func_parameters = inspect.signature(self._func).parameters.keys()

    @property
    def name(self) -> str:
        return self._func.__name__

    @property
    def description(self) -> str | None:
        return self._func.__doc__

    @property
    def parameters(self) -> dict[str, Any]:
        schema = deepcopy(self._model.schema())
        schema.pop("additionalProperties", None)
        schema.pop("title", None)
        # Pydantic adds extra parameters to the schema for the function
        # https://docs.pydantic.dev/latest/usage/validation_decorator/#model-fields-and-reserved-arguments
        schema["properties"] = {
            k: v for k, v in schema["properties"].items() if k in self._func_parameters
        }
        return schema

    def parse(self, arguments: str) -> FunctionCall[T]:
        args = self._model.parse_raw(arguments).dict(include=self._func_parameters)
        return FunctionCall(self._func, **args)


def is_union_type(type_: type) -> bool:
    type_ = get_origin(type_) or type_
    return type_ is Union or type_ is types.UnionType  # noqa: E721


P = ParamSpec("P")
R = TypeVar("R")


class PromptFunction(Generic[P, R]):
    def __init__(
        self,
        parameters: Sequence[inspect.Parameter],
        return_type: type[R],
        template: str,
        functions: list[Callable[..., Any]] | None = None,
    ):
        self._signature = inspect.Signature(
            parameters=parameters,
            return_annotation=return_type,
        )
        self._template = template
        self._functions = functions or []

        self._return_types = (
            get_args(self._signature.return_annotation)
            if is_union_type(self._signature.return_annotation)
            else [self._signature.return_annotation]
        )
        self._function_schemas: list[BaseFunctionSchema[R]] = []
        for function in self._functions:
            function_call_function_schema = FunctionCallFunctionSchema(function)
            # TODO: Make type anotations validate that `FunctionCall[<function return type>]` is included in `R`
            # TODO: This would type narrow `function_call_function_schema` to match `BaseFunctionSchema[R]`
            self._function_schemas.append(function_call_function_schema)  # type: ignore[arg-type]
        for return_type in self._return_types:
            # TODO: Skip str here. Use message for str type rather than function call
            return_type_origin = get_origin(return_type) or return_type
            if issubclass(return_type_origin, FunctionCall):
                continue
            if issubclass(return_type_origin, BaseModel):
                self._function_schemas.append(
                    BaseModelFunctionSchema(return_type_origin)
                )
            else:
                self._function_schemas.append(AnyFunctionSchema(return_type))

        self._function_schema_by_name = {
            function_schema.name: function_schema
            for function_schema in self._function_schemas
        }

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        bound_args = self._signature.bind(*args, **kwargs)
        bound_args.apply_defaults()

        response: dict[str, Any] = openai.ChatCompletion.create(  # type: ignore[no-untyped-call]
            model="gpt-3.5-turbo-0613",
            messages=[
                {
                    "role": "user",
                    "content": self._template.format(**bound_args.arguments),
                },
            ],
            functions=[
                function_schema.dict() for function_schema in self._function_schemas
            ],
            temperature=0,
        )
        response_message = response["choices"][0]["message"]
        function_name: str = response_message["function_call"]["name"]
        function_schema = self._function_schema_by_name[function_name]
        return function_schema.parse(response_message["function_call"]["arguments"])


def prompt(
    functions: list[Callable[..., Any]] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        if func.__doc__ is None:
            raise ValueError("Function must have a docstring")

        func_signature = inspect.Signature.from_callable(func)
        prompt_function = PromptFunction[P, R](
            parameters=list(func_signature.parameters.values()),
            return_type=func_signature.return_annotation,
            template=func.__doc__,
            functions=functions,
        )
        return update_wrapper(prompt_function, func)

    return decorator
