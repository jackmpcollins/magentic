from typing import Any, Callable, Iterable, ParamSpec, TypeVar

from magentic.chat_model.base import AssistantMessage, Message, UserMessage
from magentic.chat_model.openai_chat_model import OpenaiChatModel
from magentic.prompt_function import PromptFunction

P = ParamSpec("P")
Self = TypeVar("Self", bound="Chat")


class Chat:
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
        prompt: PromptFunction[P, Any],
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
        return type(self)(
            messages=[*self._messages, message],
            functions=self._functions,
            output_types=self._output_types,
            model=self._model,
        )

    def add_user_message(self: Self, content: str) -> Self:
        return self.add_message(UserMessage(content=content))

    def submit(self: Self) -> Self:
        """Request an LLM message."""
        output_message: AssistantMessage[Any] = self._model.complete(
            messages=self._messages,
            functions=self._functions,
            output_types=self._output_types,
        )
        return self.add_message(output_message)
