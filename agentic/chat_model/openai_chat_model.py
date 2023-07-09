import inspect
import json
from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
from typing import Any, Callable, Generic, Iterable, TypeVar

import openai
from pydantic import BaseModel, validate_arguments
from pydantic.generics import GenericModel

from agentic.chat_model.base import (
    AssistantMessage,
    FunctionCallMessage,
    FunctionResultMessage,
    Message,
    UserMessage,
)
from agentic.function_call import FunctionCall
from agentic.typing import is_origin_subclass

T = TypeVar("T")


class Output(GenericModel, Generic[T]):
    value: T


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
    def parse_args(self, arguments: str) -> T:
        ...

    def parse_args_to_message(self, arguments: str) -> AssistantMessage[T]:
        return AssistantMessage(self.parse_args(arguments))

    @abstractmethod
    def serialize_args(self, value: T) -> str:
        ...


class AnyFunctionSchema(BaseFunctionSchema[T], Generic[T]):
    def __init__(self, output_type: type[T]):
        self._output_type = output_type
        # https://github.com/python/mypy/issues/14458
        self._model = Output[output_type]  # type: ignore[valid-type]

    @property
    def name(self) -> str:
        return f"return_{self._output_type.__name__.lower()}"

    @property
    def parameters(self) -> dict[str, Any]:
        model_schema = self._model.schema().copy()
        model_schema.pop("title", None)
        model_schema.pop("description", None)
        return model_schema

    def parse_args(self, arguments: str) -> T:
        return self._model.parse_raw(arguments).value

    def serialize_args(self, value: T) -> str:
        return self._model(value=value).json()


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

    def parse_args(self, arguments: str) -> BaseModelT:
        return self._model.parse_raw(arguments)

    def serialize_args(self, value: BaseModelT) -> str:
        return value.json()


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

    def parse_args(self, arguments: str) -> FunctionCall[T]:
        args = self._model.parse_raw(arguments).dict(include=self._func_parameters)
        return FunctionCall(self._func, **args)

    def parse_args_to_message(self, arguments: str) -> FunctionCallMessage[T]:
        return FunctionCallMessage(self.parse_args(arguments))

    def serialize_args(self, value: FunctionCall[T]) -> str:
        return json.dumps(value.arguments)


def function_schema_for_type(type_: type[T]) -> BaseFunctionSchema[T]:
    """Create a FunctionSchema for the given type."""
    if is_origin_subclass(type_, BaseModel):
        return BaseModelFunctionSchema(type_)

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

        if isinstance(message.content, FunctionCall):
            function_schema = FunctionCallFunctionSchema(message.content.func)
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
            "name": FunctionCallFunctionSchema(message.content.func).name,
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
        messages: list[Message],
        functions: list[Callable[..., FuncR]] | None = None,
        output_types: Iterable[type[R]] = (str,),  # type: ignore[assignment]
    ) -> FunctionCallMessage[FuncR] | AssistantMessage[R]:
        """Request an LLM message."""
        function_schemas: list[
            FunctionCallFunctionSchema[FuncR] | BaseFunctionSchema[R]
        ] = []
        for function in functions or []:
            function_schemas.append(FunctionCallFunctionSchema(function))
        for output_type in output_types:
            if issubclass(output_type, str):
                pass
            else:
                function_schemas.append(function_schema_for_type(output_type))

        includes_str_output_type = any(issubclass(cls, str) for cls in output_types)

        # `openai.ChatCompletion.create` doesn't accept `None`
        # so only pass function args if there are functions
        function_args: dict[str, Any] = {}
        if function_schemas:
            function_args["functions"] = [schema.dict() for schema in function_schemas]
        if len(function_schemas) == 1 and not includes_str_output_type:
            # Force the model to call the function
            function_args["function_call"] = {"name": function_schemas[0].name}

        response: dict[str, Any] = openai.ChatCompletion.create(  # type: ignore[no-untyped-call]
            model=self._model,
            messages=[message_to_openai_message(m) for m in messages],
            temperature=self._temperature,
            **function_args,
        )
        response_message = response["choices"][0]["message"]

        if response_message.get("function_call"):
            function_schema_by_name = {
                function_schema.name: function_schema
                for function_schema in function_schemas
            }
            function_name = response_message["function_call"]["name"]
            function_schema = function_schema_by_name[function_name]
            message = function_schema.parse_args_to_message(
                response_message["function_call"]["arguments"]
            )
            return message

        if not includes_str_output_type:
            raise ValueError(
                "String was returned by model but not expected."
                " You may need to update your prompt to encourage the model to return a specific type."
            )

        return AssistantMessage(response_message["content"])  # type: ignore[return-value]
