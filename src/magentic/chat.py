import inspect
from typing import Any, Callable, Iterable, ParamSpec, TypeVar

from magentic.chat_model.message import (
    AssistantMessage,
    FunctionResultMessage,
    Message,
    UserMessage,
)
from magentic.chat_model.openai_chat_model import OpenaiChatModel
from magentic.function_call import FunctionCall
from magentic.prompt_function import BasePromptFunction

P = ParamSpec("P")
Self = TypeVar("Self", bound="Chat")


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
        messages: list[Message[Any]] | None = None,
        functions: Iterable[Callable[..., Any]] | None = None,
        output_types: Iterable[type[Any]] | None = None,
        model: OpenaiChatModel | None = None,
    ):
        self._messages = list(messages) if messages else []
        self._functions = list(functions) if functions else []
        self._output_types = list(output_types) if output_types else [str]
        self._model = model if model else OpenaiChatModel()

    @classmethod
    def from_prompt(
        cls: type[Self],
        prompt: BasePromptFunction[P, Any],
        *args: P.args,
        **kwargs: P.kwargs
    ) -> Self:
        """Create a chat from a prompt function."""
        return cls(
            messages=[UserMessage(content=prompt.format(*args, **kwargs))],
            functions=prompt.functions,
            output_types=prompt.return_types,
            model=prompt.model,
        )

    @property
    def messages(self) -> list[Message[Any]]:
        return self._messages.copy()

    def add_message(self: Self, message: Message[Any]) -> Self:
        """Add a message to the chat."""
        return type(self)(
            messages=[*self._messages, message],
            functions=self._functions,
            output_types=self._output_types,
            model=self._model,
        )

    def add_user_message(self: Self, content: str) -> Self:
        """Add a user message to the chat."""
        return self.add_message(UserMessage(content=content))

    def add_assistant_message(self: Self, content: Any) -> Self:
        """Add an assistant message to the chat."""
        return self.add_message(AssistantMessage(content=content))

    def submit(self: Self) -> Self:
        """Request an LLM message to be added to the chat."""
        output_message: AssistantMessage[Any] = self._model.complete(
            messages=self._messages,
            functions=self._functions,
            output_types=self._output_types,
        )
        return self.add_message(output_message)

    async def asubmit(self: Self) -> Self:
        """Async version of `submit`."""
        output_message: AssistantMessage[Any] = await self._model.acomplete(
            messages=self._messages,
            functions=self._functions,
            output_types=self._output_types,
        )
        return self.add_message(output_message)

    def exec_function_call(self: Self) -> Self:
        """If the last message is a function call, execute it and add the result."""
        last_message = self._messages[-1]
        if not isinstance(last_message.content, FunctionCall):
            msg = "Last message is not a function call."
            raise TypeError(msg)

        return self.add_message(
            FunctionResultMessage.from_function_call(last_message.content)
        )

    async def aexec_function_call(self: Self) -> Self:
        """Async version of `exec_function_call`."""
        last_message = self._messages[-1]
        if not isinstance(last_message.content, FunctionCall):
            msg = "Last message is not a function call."
            raise TypeError(msg)

        output = last_message.content()
        if inspect.isawaitable(output):
            output = await output

        return self.add_message(
            FunctionResultMessage(content=output, function_call=last_message.content)
        )
