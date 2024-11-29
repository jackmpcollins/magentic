from abc import ABC, abstractmethod
from collections.abc import Awaitable
from typing import (
    Annotated,
    Any,
    Generic,
    Literal,
    NamedTuple,
    TypeVar,
    cast,
    get_origin,
    overload,
)

from pydantic import BaseModel, Field, PrivateAttr
from typing_extensions import Self

from magentic.function_call import FunctionCall

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


class Message(BaseModel, Generic[ContentT], ABC):
    """A message sent to or from an LLM chat model."""

    content: ContentT

    def __init__(self, content: ContentT, **data: Any):
        super().__init__(content=content, **data)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.content!r})"

    @abstractmethod
    def format(self, **kwargs: Any) -> "Message[Any]":
        """Format the message using the provided substitutions."""
        raise NotImplementedError


class _RawMessage(Message[ContentT], Generic[ContentT]):
    """A message that gets passed directly as a `message` object to the LLM provider.

    The content of this message should be a dict/object that matches the format
    expected by the LLM provider's Python client.
    """

    def __init__(self, content: ContentT, **data: Any):
        super().__init__(content=content, **data)

    # TODO: Add Usage to _RawMessage

    def format(self, **kwargs: Any) -> "_RawMessage[ContentT]":
        del kwargs
        return _RawMessage(self.content)


class SystemMessage(Message[str]):
    """A message to the LLM to guide the whole chat."""

    role: Literal["system"] = "system"

    def __init__(self, content: str, **data: Any):
        super().__init__(content=content, **data)

    def format(self, **kwargs: Any) -> "SystemMessage":
        return SystemMessage(self.content.format(**kwargs))


class UserMessage(Message[str]):
    """A message sent by a user to an LLM chat model."""

    role: Literal["user"] = "user"

    def __init__(self, content: str, **data: Any):
        super().__init__(content=content, **data)

    def format(self, **kwargs: Any) -> "UserMessage":
        return UserMessage(self.content.format(**kwargs))


class Usage(NamedTuple):
    """Usage statistics for the LLM request."""

    input_tokens: int
    output_tokens: int


class AssistantMessage(Message[ContentT], Generic[ContentT]):
    """A message received from an LLM chat model."""

    role: Literal["assistant"] = "assistant"
    _usage_ref: list[Usage] | None = PrivateAttr(None)

    def __init__(self, content: ContentT, **data: Any):
        super().__init__(content=content, **data)

    @classmethod
    def _with_usage(cls, content: ContentT, usage_ref: list[Usage]) -> Self:
        message = cls(content)
        message._usage_ref = usage_ref
        return message

    @property
    def usage(self) -> Usage | None:
        if self._usage_ref:
            return self._usage_ref[0]
        return None

    @overload
    def format(
        self: "AssistantMessage[Placeholder[T]]", **kwargs: Any
    ) -> "AssistantMessage[T]": ...

    @overload
    def format(self: "AssistantMessage[T]", **kwargs: Any) -> "AssistantMessage[T]": ...

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


class ToolResultMessage(Message[ContentT], Generic[ContentT]):
    """A message containing the result of a tool call."""

    role: Literal["tool"] = "tool"
    tool_call_id: str

    def __init__(self, content: ContentT, tool_call_id: str, **data: Any):
        super().__init__(content=content, tool_call_id=tool_call_id, **data)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.content!r}, {self.tool_call_id=!r})"

    def format(self, **kwargs: Any) -> "ToolResultMessage[ContentT]":
        del kwargs
        return ToolResultMessage(self.content, self.tool_call_id)


class FunctionResultMessage(ToolResultMessage[ContentT], Generic[ContentT]):
    """A message containing the result of a function call."""

    _function_call: FunctionCall[Awaitable[ContentT]] | FunctionCall[ContentT]

    @overload
    def __init__(
        self,
        content: ContentT,
        function_call: FunctionCall[Awaitable[ContentT]],
        **data: Any,
    ): ...

    @overload
    def __init__(
        self, content: ContentT, function_call: FunctionCall[ContentT], **data: Any
    ): ...

    def __init__(
        self,
        content: ContentT,
        function_call: FunctionCall[Awaitable[ContentT]] | FunctionCall[ContentT],
        **data: Any,
    ):
        super().__init__(content=content, tool_call_id=function_call._unique_id, **data)
        self._function_call = function_call

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.content!r}, {self._function_call!r})"

    @property
    def function_call(
        self,
    ) -> FunctionCall[Awaitable[ContentT]] | FunctionCall[ContentT]:
        return self._function_call

    def format(self, **kwargs: Any) -> "FunctionResultMessage[ContentT]":
        del kwargs
        return FunctionResultMessage(self.content, self._function_call)


AnyMessage = Annotated[
    # Do not include FunctionResultMessage which also uses "tool" role
    SystemMessage | UserMessage | AssistantMessage[Any] | ToolResultMessage[Any],
    Field(discriminator="role"),
]
"""Union of all message types."""
