from typing import Generic, TypeVar

from magentic.function_call import FunctionCall

T = TypeVar("T")


class Message(Generic[T]):
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


class UserMessage(Message[str]):
    """A message sent by a user to an LLM chat model."""


class AssistantMessage(Message[T], Generic[T]):
    """A message received from an LLM chat model."""


class FunctionResultMessage(Message[T], Generic[T]):
    """A message containing the result of a function call."""

    def __init__(self, content: T, function_call: FunctionCall[T]):
        super().__init__(content)
        self._function_call = function_call

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.content!r}, {self._function_call!r})"

    @property
    def function_call(self) -> FunctionCall[T]:
        return self._function_call

    @classmethod
    def from_function_call(
        cls, function_call: FunctionCall[T]
    ) -> "FunctionResultMessage[T]":
        """Create a message containing the result of a function call."""
        return cls(
            content=function_call(),
            function_call=function_call,
        )


# TODO: Delete this class?
class FunctionCallMessage(AssistantMessage[FunctionCall[T]], Generic[T]):
    """A message containing a function call."""

    def get_result(self) -> FunctionResultMessage[T]:
        """Call the function and return a message containing the result."""
        return FunctionResultMessage.from_function_call(self.content)
