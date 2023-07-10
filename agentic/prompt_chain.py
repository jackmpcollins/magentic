import inspect
from functools import update_wrapper
from typing import Any, Callable, ParamSpec, TypeVar

from agentic.chat import Chat
from agentic.chat_model.base import FunctionCallMessage
from agentic.prompt_function import PromptFunction

P = ParamSpec("P")
R = TypeVar("R")


def prompt_chain(
    template: str | None = None,
    functions: list[Callable[..., Any]] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Converts a Python function to an LLM query, auto-resolving function calls."""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        if (_template := template or inspect.getdoc(func)) is None:
            raise ValueError(
                "`template` argument must be provided if function has no docstring"
            )

        func_signature = inspect.signature(func)
        prompt_function = PromptFunction[P, R](
            template=_template,
            parameters=list(func_signature.parameters.values()),
            return_type=func_signature.return_annotation,
            functions=functions,
        )

        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            chat = Chat.from_prompt(prompt_function, *args, **kwargs).submit()
            while isinstance(chat.messages[-1], FunctionCallMessage):
                function_result_message = chat.messages[-1].get_result()
                chat = chat.add_message(function_result_message).submit()
            return chat.messages[-1].content

        return update_wrapper(wrapper, func)

    return decorator
