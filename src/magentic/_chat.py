import inspect
from collections.abc import Callable, Iterable, Sequence
from typing import Any, ParamSpec

from typing_extensions import Self, deprecated

from magentic.backend import get_chat_model
from magentic.chat_model.base import ChatModel
from magentic.chat_model.message import (
    AssistantMessage,
    FunctionResultMessage,
    Message,
    SystemMessage,
    UserMessage,
    UserMessageContentBlock,
)
from magentic.function_call import (
    AsyncParallelFunctionCall,
    FunctionCall,
    ParallelFunctionCall,
)
from magentic.prompt_function import BasePromptFunction
from magentic.streaming import async_iter, azip

P = ParamSpec("P")


class Chat:
    """A chat with an LLM chat model.

    Examples
    --------
    >>> chat = Chat().add_user_message("Hello")
    >>> chat.messages
    [UserMessage('Hello')]
    >>> chat = chat.submit()
    >>> chat.messages
    [UserMessage('Hello'), AssistantMessage('Hello! How can I assist you today?')]
    """

    def __init__(
        self,
        messages: Sequence[Message[Any]] | None = None,
        *,
        functions: Iterable[Callable[..., Any]] | None = None,
        output_types: Iterable[type[Any]] | None = None,
        model: ChatModel | None = None,
    ):
        self._messages = list(messages) if messages else []
        self._functions = list(functions) if functions else []
        self._output_types = list(output_types) if output_types else [str]
        self._model = model

    @classmethod
    @deprecated(
        "Chat.from_prompt will be removed in a future version."
        " Instead, use the regular init method, `Chat(messages, functions, output_types, model)`."
    )
    def from_prompt(
        cls: type[Self],
        prompt: BasePromptFunction[P, Any],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Self:
        """Create a chat from a prompt function."""
        return cls(
            messages=[UserMessage(content=prompt.format(*args, **kwargs))],
            functions=prompt.functions,
            output_types=prompt.return_types,
            model=prompt._model,  # Keep `None` value if unset
        )

    @property
    def messages(self) -> list[Message[Any]]:
        return self._messages.copy()

    @property
    def last_message(self) -> Message[Any]:
        return self._messages[-1]

    @property
    def model(self) -> ChatModel:
        return self._model or get_chat_model()

    def add_message(self, message: Message[Any]) -> Self:
        """Add a message to the chat."""
        return type(self)(
            messages=[*self._messages, message],
            functions=self._functions,
            output_types=self._output_types,
            model=self._model,  # Keep `None` value if unset
        )

    def add_system_message(self, content: str) -> Self:
        """Add a system message to the chat."""
        return self.add_message(SystemMessage(content=content))

    def add_user_message(
        self, content: str | Sequence[str | UserMessageContentBlock]
    ) -> Self:
        """Add a user message to the chat."""
        return self.add_message(UserMessage(content=content))

    def add_assistant_message(self, content: Any) -> Self:
        """Add an assistant message to the chat."""
        return self.add_message(AssistantMessage(content=content))

    # TODO: Allow restricting functions and/or output types here
    def submit(self) -> Self:
        """Request an LLM message to be added to the chat."""
        output_message: AssistantMessage[Any] = self.model.complete(
            messages=self._messages,
            functions=self._functions,
            output_types=self._output_types,
        )
        return self.add_message(output_message)

    async def asubmit(self) -> Self:
        """Async version of `submit`."""
        output_message: AssistantMessage[Any] = await self.model.acomplete(
            messages=self._messages,
            functions=self._functions,
            output_types=self._output_types,
        )
        return self.add_message(output_message)

    # TODO: Add optional error handling to this method, with param to toggle
    def exec_function_call(self) -> Self:
        """If the last message is a function call, execute it and add the result."""
        if isinstance(self.last_message.content, FunctionCall):
            function_call = self.last_message.content
            result = function_call()
            return self.add_message(
                FunctionResultMessage(content=result, function_call=function_call)
            )

        if isinstance(self.last_message.content, ParallelFunctionCall):
            parallel_function_call = self.last_message.content
            chat = self
            for result, function_call in zip(
                parallel_function_call(), parallel_function_call, strict=True
            ):
                chat = chat.add_message(
                    FunctionResultMessage(content=result, function_call=function_call)
                )
            return chat

        msg = "Last message is not a function call."
        raise TypeError(msg)

    async def aexec_function_call(self) -> Self:
        """Async version of `exec_function_call`."""
        if isinstance(self.last_message.content, FunctionCall):
            function_call = self.last_message.content
            result = function_call()
            if inspect.isawaitable(result):
                result = await result
            return self.add_message(
                FunctionResultMessage(content=result, function_call=function_call)
            )

        if isinstance(self.last_message.content, AsyncParallelFunctionCall):
            async_parallel_function_call = self.last_message.content
            chat = self
            async for result, function_call in azip(
                async_iter(await async_parallel_function_call()),
                async_parallel_function_call,
            ):
                chat = chat.add_message(
                    FunctionResultMessage(content=result, function_call=function_call)
                )
            return chat

        msg = "Last message is not a function call."
        raise TypeError(msg)
