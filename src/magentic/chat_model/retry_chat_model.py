from collections.abc import Callable, Iterable, Sequence
from functools import partial, singledispatch, singledispatchmethod
from typing import Any, TypeVar

from magentic.chat_model.base import ChatModel, OutputT, ToolSchemaParseError
from magentic.chat_model.message import AssistantMessage, Message, ToolResultMessage
from magentic.logger import logfire

ExceptionT = TypeVar("ExceptionT", bound=Exception)


class RetryChatModel(ChatModel):
    """Wraps another ChatModel to add LLM-assisted retries."""

    def __init__(
        self,
        chat_model: ChatModel,
        *,
        max_retries: int,
    ):
        self._chat_model = chat_model
        self._max_retries = max_retries
        # A separate dispatcher per instance prevents registrations leaking.
        self._exception_handlers = singledispatch(
            lambda error: self._make_retry_messages(error)
        )
        self._exception_handlers.register(
            ToolSchemaParseError, self._make_retry_messages
        )

    def register_handler(
        self, exception_type: type[ExceptionT]
    ) -> Callable[
        [Callable[["RetryChatModel", ExceptionT], Sequence[Message[Any]]]],
        Callable[["RetryChatModel", ExceptionT], Sequence[Message[Any]]],
    ]:
        """Register a synchronous completion-error handler on this instance.

        The handler receives this model and the error, and returns messages to
        append before retrying. Subclasses use the most specific registration.
        Registering the same type again replaces its handler on this instance.
        """
        if not isinstance(exception_type, type) or not issubclass(
            exception_type, Exception
        ):
            msg = "Handlers must target an Exception subclass, not BaseException."
            raise TypeError(msg)

        def decorator(
            handler: Callable[["RetryChatModel", ExceptionT], Sequence[Message[Any]]],
        ) -> Callable[["RetryChatModel", ExceptionT], Sequence[Message[Any]]]:
            self._exception_handlers.register(exception_type, partial(handler, self))
            return handler

        return decorator

    def _retry_exception_types(self) -> tuple[type[Exception], ...]:
        return tuple(
            exception_type
            for exception_type in self._exception_handlers.registry
            if issubclass(exception_type, Exception)
        )

    # TODO: Add exception base class for those with output_message attribute
    @singledispatchmethod
    def _make_retry_messages(self, error: Exception) -> list[Message[Any]]:
        raise NotImplementedError

    # TODO: Catch UnknownToolError here
    @_make_retry_messages.register
    def _(self, error: ToolSchemaParseError) -> list[Message[Any]]:
        return [
            error.output_message,
            ToolResultMessage(
                content=str(error.validation_error), tool_call_id=error.tool_call_id
            ),
        ]

    def complete(
        self,
        messages: Iterable[Message[Any]],
        functions: Iterable[Callable[..., Any]] | None = None,
        output_types: Iterable[type[OutputT]] | None = None,
        *,
        stop: list[str] | None = None,
    ) -> AssistantMessage[OutputT]:
        """Request an LLM message."""
        with logfire.span(
            "LLM-assisted retries enabled. Max {max_retries}",
            max_retries=self._max_retries,
        ):
            messages = list(messages)
            num_retry = 0
            while True:
                try:
                    message = self._chat_model.complete(
                        messages=messages,
                        functions=functions,
                        output_types=output_types,
                        stop=stop,
                    )
                except self._retry_exception_types() as e:
                    if num_retry >= self._max_retries:
                        raise
                    messages += self._exception_handlers(e)
                else:
                    return message

                num_retry += 1
                # TODO: Also log using Python logger
                logfire.warn(
                    "Retrying Chat Completion. Attempt {num_retry}",
                    num_retry=num_retry,
                )

    async def acomplete(
        self,
        messages: Iterable[Message[Any]],
        functions: Iterable[Callable[..., Any]] | None = None,
        output_types: Iterable[type[OutputT]] | None = None,
        *,
        stop: list[str] | None = None,
    ) -> AssistantMessage[OutputT]:
        """Async version of `complete`."""
        with logfire.span(
            "LLM-assisted retries enabled. Max {max_retries}",
            max_retries=self._max_retries,
        ):
            messages = list(messages)
            num_retry = 0
            while True:
                try:
                    message = await self._chat_model.acomplete(
                        messages=messages,
                        functions=functions,
                        output_types=output_types,
                        stop=stop,
                    )
                except self._retry_exception_types() as e:
                    if num_retry >= self._max_retries:
                        raise
                    messages += self._exception_handlers(e)
                else:
                    return message

                num_retry += 1
                # TODO: Also log using Python logger
                logfire.warn(
                    "Retrying Chat Completion. Attempt {num_retry}",
                    num_retry=num_retry,
                )
