from abc import ABC, abstractmethod
from typing import Any, Awaitable, Generic, TypeVar, cast, overload

from magentic.function_call import FunctionCall

T = TypeVar("T")


class Message(Generic[T], ABC):
    """A message sent to or from an LLM chat model."""

    def __init__(self, content: T):
        self._content = content

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return type(self) is type(other) and self.content == other.content

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.content!r})"

    @property
    def content(self) -> T:
        return self._content

    @abstractmethod
    def with_content(self, content: T) -> "Message[T]":
        raise NotImplementedError

    @abstractmethod
    def format(self, **kwargs: Any) -> "Message[T]":
        raise NotImplementedError


class SystemMessage(Message[str]):
    """A message to the LLM to guide the whole chat."""

    def with_content(self, content: str) -> "SystemMessage":
        return SystemMessage(content)

    def format(self, **kwargs: Any) -> "SystemMessage":
        return self.with_content(self.content.format(**kwargs))


class UserMessage(Message[str]):
    """A message sent by a user to an LLM chat model."""

    def with_content(self, content: str) -> "UserMessage":
        return UserMessage(content)

    def format(self, **kwargs: Any) -> "UserMessage":
        return self.with_content(self.content.format(**kwargs))


class AssistantMessage(Message[T], Generic[T]):
    """A message received from an LLM chat model."""

    def with_content(self, content: T) -> "AssistantMessage[T]":
        return AssistantMessage(content)

    def format(self, **kwargs: Any) -> "AssistantMessage[T]":
        if isinstance(self.content, str):
            # Cast back to more general type `T` to satisfy mypy
            content = cast(T, self.content.format(**kwargs))
            return self.with_content(content)
        return self


class FunctionResultMessage(Message[T], Generic[T]):
    """A message containing the result of a function call."""

    @overload
    def __init__(self, content: T, function_call: FunctionCall[T]):
        ...

    @overload
    def __init__(self, content: T, function_call: FunctionCall[Awaitable[T]]):
        ...

    def __init__(
        self, content: T, function_call: FunctionCall[T] | FunctionCall[Awaitable[T]]
    ):
        super().__init__(content)
        self._function_call = function_call

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.content!r}, {self._function_call!r})"

    @property
    def function_call(self) -> FunctionCall[T] | FunctionCall[Awaitable[T]]:
        return self._function_call

    def with_content(self, content: T) -> "FunctionResultMessage[T]":
        return FunctionResultMessage(content, self._function_call)

    def format(self, **kwargs: Any) -> "FunctionResultMessage[T]":
        if isinstance(self.content, str):
            # Cast back to more general type `T` to sa
            content = cast(T, self.content.format(**kwargs))
            return self.with_content(content)
        return self

    @classmethod
    def from_function_call(
        cls, function_call: FunctionCall[T]
    ) -> "FunctionResultMessage[T]":
        """Create a message containing the result of a function call."""
        return cls(
            content=function_call(),
            function_call=function_call,
        )

    @classmethod
    async def afrom_function_call(
        cls, function_call: FunctionCall[Awaitable[T]]
    ) -> "FunctionResultMessage[T]":
        """Async version of `from_function_call`."""
        return cls(
            content=await function_call(),
            function_call=function_call,
        )
