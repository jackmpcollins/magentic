from abc import ABC, abstractmethod
from typing import Awaitable, Callable, Generic, TypeVar, overload

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


class SystemMessage(Message[str]):
    """A message to the LLM to guide the whole chat."""

    def with_content(self, content: str) -> "SystemMessage":
        return SystemMessage(content)


class UserMessage(Message[str]):
    """A message sent by a user to an LLM chat model."""

    def with_content(self, content: str) -> "UserMessage":
        return UserMessage(content)


class AssistantMessage(Message[T], Generic[T]):
    """A message received from an LLM chat model."""

    def with_content(self, content: T) -> "AssistantMessage[T]":
        return AssistantMessage(content)


class FunctionResultMessage(Message[T], Generic[T]):
    """A message containing the result of a function call."""

    @overload
    def __init__(self, content: T, function: Callable[..., Awaitable[T]]):
        ...

    @overload
    def __init__(self, content: T, function: Callable[..., T]):
        ...

    def __init__(
        self, content: T, function: Callable[..., Awaitable[T]] | Callable[..., T]
    ):
        super().__init__(content)
        self._function = function

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.content!r}, {self._function!r})"

    @property
    def function(self) -> Callable[..., Awaitable[T]] | Callable[..., T]:
        return self._function

    def with_content(self, content: T) -> "FunctionResultMessage[T]":
        return FunctionResultMessage(content, self._function)
