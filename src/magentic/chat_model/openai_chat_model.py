import inspect
import json
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Generic, Iterable, TypeVar

import openai
from pydantic import BaseModel, create_model

from magentic.chat_model.base import (
    AssistantMessage,
    FunctionCallMessage,
    FunctionResultMessage,
    Message,
    UserMessage,
)
from magentic.function_call import FunctionCall
from magentic.typing import is_origin_subclass, name_type

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
    def parse_args(self, arguments: str) -> T:
        ...

    def parse_args_to_message(self, arguments: str) -> AssistantMessage[T]:
        return AssistantMessage(self.parse_args(arguments))

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

    def parse_args(self, arguments: str) -> T:
        return self._model.model_validate_json(arguments).value

    def serialize_args(self, value: T) -> str:
        return self._model(value=value).model_dump_json()


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

    def parse_args(self, arguments: str) -> BaseModelT:
        return self._model.model_validate_json(arguments)

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

    def parse_args(self, arguments: str) -> FunctionCall[T]:
        args = self._model.model_validate_json(arguments).model_dump()
        return FunctionCall(self._func, **args)

    def parse_args_to_message(self, arguments: str) -> FunctionCallMessage[T]:
        return FunctionCallMessage(self.parse_args(arguments))

    def serialize_args(self, value: FunctionCall[T]) -> str:
        return json.dumps(value.arguments)


def function_schema_for_type(type_: type[T]) -> BaseFunctionSchema[T]:
    """Create a FunctionSchema for the given type."""
    if is_origin_subclass(type_, BaseModel):
        return BaseModelFunctionSchema(type_)  # type: ignore[return-value]

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

        function_schemas: list[
            FunctionCallFunctionSchema[FuncR] | BaseFunctionSchema[R]
        ] = []
        for function in functions or []:
            function_schemas.append(FunctionCallFunctionSchema(function))
        for output_type in output_types:
            if issubclass(output_type, str):
                continue
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
                "String was returned by model but not expected. You may need to update"
                " your prompt to encourage the model to return a specific type."
            )

        return AssistantMessage(response_message["content"])
