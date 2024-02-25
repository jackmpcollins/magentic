import types
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from contextvars import ContextVar
from typing import Any, TypeVar, overload

from magentic.chat_model.message import (
    AssistantMessage,
    Message,
)
from magentic.function_call import FunctionCall

R = TypeVar("R")
FuncR = TypeVar("FuncR")

_chat_model_context: ContextVar["ChatModel | None"] = ContextVar(
    "chat_model", default=None
)


class StructuredOutputError(Exception):
    """Raised when the LLM output could not be parsed."""


class ChatModel(ABC):
    """An LLM chat model."""

    @overload
    @abstractmethod
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
    @abstractmethod
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
    @abstractmethod
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
    @abstractmethod
    def complete(
        self,
        messages: Iterable[Message[Any]],
        functions: Iterable[Callable[..., FuncR]],
        output_types: Iterable[type[R]],
        *,
        stop: list[str] | None = ...,
    ) -> AssistantMessage[FunctionCall[FuncR]] | AssistantMessage[R]:
        ...

    @abstractmethod
    def complete(
        self,
        messages: Iterable[Message[Any]],
        functions: Iterable[Callable[..., FuncR]] | None = None,
        output_types: Iterable[type[R | str]] | None = None,
        *,
        stop: list[str] | None = None,
    ) -> (
        AssistantMessage[FunctionCall[FuncR]]
        | AssistantMessage[R]
        | AssistantMessage[str]
    ):
        """Request an LLM message."""
        ...

    @overload
    @abstractmethod
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
    @abstractmethod
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
    @abstractmethod
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
    @abstractmethod
    async def acomplete(
        self,
        messages: Iterable[Message[Any]],
        functions: Iterable[Callable[..., FuncR]],
        output_types: Iterable[type[R]],
        *,
        stop: list[str] | None = ...,
    ) -> AssistantMessage[FunctionCall[FuncR]] | AssistantMessage[R]:
        ...

    @abstractmethod
    async def acomplete(
        self,
        messages: Iterable[Message[Any]],
        functions: Iterable[Callable[..., FuncR]] | None = None,
        output_types: Iterable[type[R | str]] | None = None,
        *,
        stop: list[str] | None = None,
    ) -> (
        AssistantMessage[FunctionCall[FuncR]]
        | AssistantMessage[R]
        | AssistantMessage[str]
    ):
        """Async version of `complete`."""
        ...

    def __enter__(self) -> None:
        self.__token = _chat_model_context.set(self)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        _chat_model_context.reset(self.__token)
