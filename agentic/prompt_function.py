from functools import wraps
from inspect import Parameter, Signature
import types
from typing import Generic, Sequence, Type, TypeVar, Union, get_args, get_origin

import openai
from pydantic import BaseModel
from pydantic.generics import GenericModel


T = TypeVar("T")


class Output(GenericModel, Generic[T]):
    value: T


class AnyFunctionSchema(Generic[T]):
    def __init__(self, return_type: Type[T]):
        self._return_type = return_type

    @property
    def name(self) -> str:
        return f"return_{self._return_type.__name__.lower()}"

    def dict(self) -> dict:
        model_schema = Output[self._return_type].schema().copy()
        model_schema.pop("title", None)
        model_schema.pop("description", None)
        return {
            "name": self.name,
            "parameters": model_schema,
        }

    def parse(self, arguments: str) -> T:
        return Output[self._return_type].parse_raw(arguments).value


TBaseModel = TypeVar("TBaseModel", bound=BaseModel)


class BaseModelFunctionSchema(Generic[TBaseModel]):
    def __init__(self, model: Type[TBaseModel]):
        self._model = model

    @property
    def name(self) -> str:
        return f"return_{self._model.__name__.lower()}"

    def dict(self) -> dict:
        model_schema = self._model.schema().copy()
        model_schema.pop("title", None)
        model_schema.pop("description", None)
        return {
            "name": self.name,
            "parameters": model_schema,
        }

    def parse(self, arguments: str) -> TBaseModel:
        return self._model.parse_raw(arguments)


class PromptFunction:
    def __init__(
        self,
        parameters: Sequence[Parameter],
        return_type: Type,
        template: str,
    ):
        self._signature = Signature(
            parameters=parameters,
            return_annotation=return_type,
        )
        self._template = template

    def __call__(self, *args, **kwargs):
        bound_args = self._signature.bind(*args, **kwargs)
        bound_args.apply_defaults()

        return_origin = get_origin(self._signature.return_annotation)
        if return_origin is Union or return_origin is types.UnionType:
            return_types = get_args(self._signature.return_annotation)
        else:
            return_types = [self._signature.return_annotation]

        function_schema_by_name = {}
        for return_type in return_types:
            # TODO: Skip str here. Use message for str type rather than function call
            if issubclass(return_type, BaseModel):
                function_schema = BaseModelFunctionSchema(return_type)
            else:
                function_schema = AnyFunctionSchema(return_type)
            function_schema_by_name[function_schema.name] = function_schema

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=[
                {
                    "role": "user",
                    "content": self._template.format(**bound_args.arguments),
                },
            ],
            functions=[
                function_schema.dict()
                for function_schema in function_schema_by_name.values()
            ],
            temperature=0,
        )
        response_message = response["choices"][0]["message"]
        function_name: str = response_message["function_call"]["name"]
        function_schema = function_schema_by_name[function_name]
        return function_schema.parse(response_message["function_call"]["arguments"])


def prompt():
    def decorator(func):
        func_signature = Signature.from_callable(func)
        return wraps(func)(
            PromptFunction(
                parameters=list(func_signature.parameters.values()),
                return_type=func_signature.return_annotation,
                template=func.__doc__,
            )
        )

    return decorator
