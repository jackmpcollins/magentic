import types
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from contextvars import ContextVar
from typing import Any, TypeVar, overload

from pydantic import ValidationError

from magentic.chat_model.message import AssistantMessage, Message
from magentic.streaming import AsyncStreamedStr, StreamedStr

R = TypeVar("R")

_chat_model_context: ContextVar["ChatModel | None"] = ContextVar(
    "chat_model", default=None
)


# TODO: Parent class with `output_message` attribute ?
class StringNotAllowedError(Exception):
    """Raised when a string is returned by the LLM but not expected."""

    _MESSAGE = (
        "A string was returned by the LLM but was not an allowed output type."
        ' Consider updating the prompt to encourage the LLM to "use the tool".'
        " Model output: {model_output!r}"
    )

    def __init__(self, output_message: Message[Any]):
        super().__init__(self._MESSAGE.format(model_output=output_message.content))
        self.output_message = output_message


class ToolSchemaParseError(Exception):
    """Raised when the LLM output could not be parsed by the tool schema."""

    _MESSAGE = (
        "Failed to parse the LLM output into the tool schema."
        " Consider making the output type more lenient or enabling retries."
        " Model output: {model_output!r}"
    )

    def __init__(
        self,
        output_message: Message[Any],
        tool_call_id: str,
        validation_error: ValidationError,
    ):
        super().__init__(self._MESSAGE.format(model_output=output_message.content))
        self.output_message = output_message
        self.tool_call_id = tool_call_id
        self.validation_error = validation_error


def validate_str_content(
    streamed_str: StreamedStr, *, allow_string_output: bool, streamed: bool
) -> StreamedStr | str:
    """Raise error if string output not expected. Otherwise return correct string type."""
    if not allow_string_output:
        model_output = streamed_str.truncate(100)
        raise StringNotAllowedError(AssistantMessage(model_output))
    if streamed:
        return streamed_str
    return str(streamed_str)


async def avalidate_str_content(
    async_streamed_str: AsyncStreamedStr, *, allow_string_output: bool, streamed: bool
) -> AsyncStreamedStr | str:
    """Async version of `validate_str_content`."""
    if not allow_string_output:
        model_output = await async_streamed_str.truncate(100)
        raise StringNotAllowedError(AssistantMessage(model_output))
    if streamed:
        return async_streamed_str
    return await async_streamed_str.to_string()


class ChatModel(ABC):
    """An LLM chat model."""

    @overload
    @abstractmethod
    def complete(
        self,
        messages: Iterable[Message[Any]],
        functions: Any = ...,
        output_types: None = ...,
        *,
        stop: list[str] | None = ...,
    ) -> AssistantMessage[str]: ...

    @overload
    @abstractmethod
    def complete(
        self,
        messages: Iterable[Message[Any]],
        functions: Any = ...,
        output_types: Iterable[type[R]] = ...,
        *,
        stop: list[str] | None = ...,
    ) -> AssistantMessage[R]: ...

    @abstractmethod
    def complete(
        self,
        messages: Iterable[Message[Any]],
        functions: Iterable[Callable[..., Any]] | None = None,
        output_types: Iterable[type[R | str]] | None = None,
        *,
        stop: list[str] | None = None,
    ) -> AssistantMessage[str] | AssistantMessage[R]:
        """Request an LLM message."""
        ...

    @overload
    @abstractmethod
    async def acomplete(
        self,
        messages: Iterable[Message[Any]],
        functions: Any = ...,
        output_types: None = ...,
        *,
        stop: list[str] | None = ...,
    ) -> AssistantMessage[str]: ...

    @overload
    @abstractmethod
    async def acomplete(
        self,
        messages: Iterable[Message[Any]],
        functions: Any = ...,
        output_types: Iterable[type[R]] = ...,
        *,
        stop: list[str] | None = ...,
    ) -> AssistantMessage[R]: ...

    @abstractmethod
    async def acomplete(
        self,
        messages: Iterable[Message[Any]],
        functions: Iterable[Callable[..., Any]] | None = None,
        output_types: Iterable[type[R | str]] | None = None,
        *,
        stop: list[str] | None = None,
    ) -> AssistantMessage[str] | AssistantMessage[R]:
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
