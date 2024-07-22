import types
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from contextvars import ContextVar
from typing import Any, TypeVar, overload

from magentic.chat_model.message import AssistantMessage, Message, UserMessage
from magentic.streaming import AsyncStreamedStr, StreamedStr

R = TypeVar("R")

_chat_model_context: ContextVar["ChatModel | None"] = ContextVar(
    "chat_model", default=None
)


class StructuredOutputError(Exception):
    """Raised when the LLM output could not be parsed."""

    def __init__(
        self, message: str, output_message: Message[Any], retry_message: Message[Any]
    ):
        super().__init__(message)
        self.output_message = output_message
        self.retry_message = retry_message


_STRING_NOT_EXPECTED_ERROR_MESSAGE = (
    "String was returned by model but not expected. You may need to update"
    " your prompt to encourage the model to return a specific type."
    " Model output: {model_output!r}"
)
# TODO: Enable users to modify this prompt. Add to settings?
_STRING_NOT_EXPECTED_RETRY_MESSAGE = "Use the tools provided. Do not return a string."


def validate_str_content(
    streamed_str: StreamedStr, *, allow_string_output: bool, streamed: bool
) -> StreamedStr | str:
    """Raise error if string output not expected. Otherwise return correct string type."""
    if not allow_string_output:
        model_output = streamed_str.truncate(100)
        raise StructuredOutputError(
            _STRING_NOT_EXPECTED_ERROR_MESSAGE.format(model_output=model_output),
            output_message=AssistantMessage(model_output),
            retry_message=UserMessage(_STRING_NOT_EXPECTED_RETRY_MESSAGE),
        )
    if streamed:
        return streamed_str
    return str(streamed_str)


async def avalidate_str_content(
    async_streamed_str: AsyncStreamedStr, *, allow_string_output: bool, streamed: bool
) -> AsyncStreamedStr | str:
    """Async version of `validate_str_content`."""
    if not allow_string_output:
        model_output = await async_streamed_str.truncate(100)
        raise StructuredOutputError(
            _STRING_NOT_EXPECTED_ERROR_MESSAGE.format(model_output=model_output),
            output_message=AssistantMessage(model_output),
            retry_message=UserMessage(_STRING_NOT_EXPECTED_RETRY_MESSAGE),
        )
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
