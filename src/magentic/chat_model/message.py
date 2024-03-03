from abc import ABC, abstractmethod
from typing import (
    Any,
    Awaitable,
    Callable,
    Generic,
    TypeVar,
    cast,
    get_origin,
    overload,
)

T = TypeVar("T")


class Placeholder(Generic[T]):
    """A placeholder for a value in a message.

    When formatting a message, the placeholder is replaced with the value.
    This is used in combination with the `@prompt`, `@promptchain`, and
    `@chatprompt` decorators to enable inserting function arguments into
    messages.
    """

    def __init__(self, type_: type[T], name: str):
        self.type_ = type_
        self.name = name

    def format(self, **kwargs: Any) -> T:
        value = kwargs[self.name]
        if not isinstance(value, get_origin(self.type_) or self.type_):
            msg = f"{self.name} must be of type {self.type_}"
            raise TypeError(msg)
        return cast(T, value)


ContentT = TypeVar("ContentT")


class Message(Generic[ContentT], ABC):
    """A message sent to or from an LLM chat model."""

    def __init__(self, content: ContentT):
        self._content = content

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return type(self) is type(other) and self.content == other.content

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.content!r})"

    @property
    def content(self) -> ContentT:
        return self._content

    @abstractmethod
    def format(self, **kwargs: Any) -> "Message[Any]":
        """Format the message using the provided substitutions."""
        raise NotImplementedError


class SystemMessage(Message[str]):
    """A message to the LLM to guide the whole chat."""

    def format(self, **kwargs: Any) -> "SystemMessage":
        return SystemMessage(self.content.format(**kwargs))


class UserMessage(Message[str]):
    """A message sent by a user to an LLM chat model."""

    def format(self, **kwargs: Any) -> "UserMessage":
        return UserMessage(self.content.format(**kwargs))


class AssistantMessage(Message[ContentT], Generic[ContentT]):
    """A message received from an LLM chat model."""

    @overload
    def format(
        self: "AssistantMessage[Placeholder[T]]", **kwargs: Any
    ) -> "AssistantMessage[T]":
        ...

    @overload
    def format(self: "AssistantMessage[T]", **kwargs: Any) -> "AssistantMessage[T]":
        ...

    def format(
        self: "AssistantMessage[Placeholder[T]] | AssistantMessage[T]", **kwargs: Any
    ) -> "AssistantMessage[T]":
        if isinstance(self.content, str):
            formatted_content = cast(T, self.content.format(**kwargs))
            return AssistantMessage(formatted_content)
        if isinstance(self.content, Placeholder):
            content = cast(Placeholder[T], self.content)
            return AssistantMessage(content.format(**kwargs))
        return AssistantMessage(self.content)


class FunctionResultMessage(Message[ContentT], Generic[ContentT]):
    """A message containing the result of a function call."""

    @overload
    def __init__(self, content: ContentT, function: Callable[..., Awaitable[ContentT]]):
        ...

    @overload
    def __init__(self, content: ContentT, function: Callable[..., ContentT]):
        ...

    def __init__(
        self,
        content: ContentT,
        function: Callable[..., Awaitable[ContentT]] | Callable[..., ContentT],
    ):
        super().__init__(content)
        self._function = function

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.content!r}, {self._function!r})"

    @property
    def function(self) -> Callable[..., Awaitable[ContentT]] | Callable[..., ContentT]:
        return self._function

    def format(self, **kwargs: Any) -> "FunctionResultMessage[ContentT]":
        del kwargs
        return FunctionResultMessage(self.content, self.function)
