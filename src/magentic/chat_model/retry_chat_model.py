from collections.abc import Callable, Iterable
from functools import singledispatchmethod
from typing import Any

from magentic.chat_model.base import ChatModel, OutputT, ToolSchemaParseError
from magentic.chat_model.message import AssistantMessage, Message, ToolResultMessage
from magentic.logger import logfire


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

    # TODO: Make this public to allow modifying error handling behavior
    # User should be able to add handlers to instance using decorator
    # e.g. `@my_retry_chat_model.exception_handler(exc_type)`
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
                # TODO: Get list of caught exceptions from _make_retry_messages registered types
                except ToolSchemaParseError as e:
                    if num_retry >= self._max_retries:
                        raise
                    messages += self._make_retry_messages(e)
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
                except ToolSchemaParseError as e:
                    if num_retry >= self._max_retries:
                        raise
                    messages += self._make_retry_messages(e)
                else:
                    return message

                num_retry += 1
                # TODO: Also log using Python logger
                logfire.warn(
                    "Retrying Chat Completion. Attempt {num_retry}",
                    num_retry=num_retry,
                )
