from abc import ABC, abstractmethod
from typing import (
    Any,
    Awaitable,
    Generic,
    TypeVar,
    cast,
    get_origin,
    overload,
)

from magentic.function_call import FunctionCall

T = TypeVar("T")


class Placeholder(Generic[T]):
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

    @overload
    @abstractmethod
    def format(self: "Message[str]", **kwargs: Any) -> "Message[str]":
        ...

    @overload
    @abstractmethod
    def format(self: "Message[Placeholder[T]]", **kwargs: Any) -> "Message[T]":
        ...

    @overload
    @abstractmethod
    def format(self: "Message[T]", **kwargs: Any) -> "Message[T]":
        ...

    @abstractmethod
    def format(
        self: "Message[str] | Message[Placeholder[T]] | Message[T]", **kwargs: Any
    ) -> "Message[str] | Message[T]":
        raise NotImplementedError


class SystemMessage(Message[str]):
    """A message to the LLM to guide the whole chat."""

    def format(self: "SystemMessage", **kwargs: Any) -> "SystemMessage":
        return SystemMessage(self.content.format(**kwargs))


class UserMessage(Message[str]):
    """A message sent by a user to an LLM chat model."""

    def format(self, **kwargs: Any) -> "UserMessage":
        return UserMessage(self.content.format(**kwargs))


class AssistantMessage(Message[ContentT], Generic[ContentT]):
    """A message received from an LLM chat model."""

    @overload
    def format(self: "AssistantMessage[str]", **kwargs: Any) -> "AssistantMessage[str]":
        ...

    @overload
    def format(
        self: "AssistantMessage[Placeholder[T]]", **kwargs: Any
    ) -> "AssistantMessage[T]":
        ...

    @overload
    def format(self: "AssistantMessage[T]", **kwargs: Any) -> "AssistantMessage[T]":
        ...

    def format(
        self: "AssistantMessage[str] | AssistantMessage[Placeholder[T]] | AssistantMessage[T]",
        **kwargs: Any,
    ) -> "AssistantMessage[str] | AssistantMessage[T]":
        if isinstance(self.content, str):
            return AssistantMessage(self.content.format(**kwargs))
        if isinstance(self.content, Placeholder):
            content = cast(Placeholder[T], self.content)
            return AssistantMessage(content.format(**kwargs))
        return cast(AssistantMessage[T], self)


class FunctionResultMessage(Message[ContentT], Generic[ContentT]):
    """A message containing the result of a function call."""

    @overload
    def __init__(self, content: ContentT, function_call: FunctionCall[ContentT]):
        ...

    @overload
    def __init__(
        self, content: ContentT, function_call: FunctionCall[Awaitable[ContentT]]
    ):
        ...

    def __init__(
        self,
        content: ContentT,
        function_call: FunctionCall[ContentT] | FunctionCall[Awaitable[ContentT]],
    ):
        super().__init__(content)
        self._function_call = function_call

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.content!r}, {self._function_call!r})"

    @property
    def function_call(
        self,
    ) -> FunctionCall[ContentT] | FunctionCall[Awaitable[ContentT]]:
        return self._function_call

    @overload
    def format(
        self: "FunctionResultMessage[str]", **kwargs: Any
    ) -> "FunctionResultMessage[str]":
        ...

    @overload
    def format(
        self: "FunctionResultMessage[Placeholder[T]]", **kwargs: Any
    ) -> "FunctionResultMessage[T]":
        ...

    @overload
    def format(
        self: "FunctionResultMessage[T]", **kwargs: Any
    ) -> "FunctionResultMessage[T]":
        ...

    def format(
        self: "FunctionResultMessage[str] | FunctionResultMessage[Placeholder[T]] | FunctionResultMessage[T]",
        **kwargs: Any,
    ) -> "FunctionResultMessage[str] | FunctionResultMessage[T]":
        if isinstance(self.content, str):
            function_call_str = cast(
                FunctionCall[str] | FunctionCall[Awaitable[str]], self._function_call
            )
            return FunctionResultMessage(
                self.content.format(**kwargs), function_call_str
            )
        if isinstance(self.content, Placeholder):
            content = cast(Placeholder[T], self.content)
            function_call_x = cast(
                FunctionCall[T] | FunctionCall[Awaitable[T]],
                self._function_call,
            )
            return FunctionResultMessage(content.format(**kwargs), function_call_x)
        return cast(FunctionResultMessage[T], self)

    @classmethod
    def from_function_call(
        cls, function_call: FunctionCall[ContentT]
    ) -> "FunctionResultMessage[ContentT]":
        """Create a message containing the result of a function call."""
        return cls(
            content=function_call(),
            function_call=function_call,
        )

    @classmethod
    async def afrom_function_call(
        cls, function_call: FunctionCall[Awaitable[ContentT]]
    ) -> "FunctionResultMessage[ContentT]":
        """Async version of `from_function_call`."""
        return cls(
            content=await function_call(),
            function_call=function_call,
        )
