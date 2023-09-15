import inspect
from functools import update_wrapper
from typing import Any, Callable, ParamSpec, TypeVar, cast

from magentic.chat import Chat
from magentic.chat_model.base import FunctionCallMessage
from magentic.chat_model.openai_chat_model import OpenaiChatModel
from magentic.prompt_function import AsyncPromptFunction, PromptFunction

P = ParamSpec("P")
R = TypeVar("R")


def prompt_chain(
    template: str,
    functions: list[Callable[..., Any]] | None = None,
    model: OpenaiChatModel | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Convert a Python function to an LLM query, auto-resolving function calls."""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        func_signature = inspect.signature(func)

        if inspect.iscoroutinefunction(func):
            async_prompt_function = AsyncPromptFunction[P, R](
                template=template,
                parameters=list(func_signature.parameters.values()),
                return_type=func_signature.return_annotation,
                functions=functions,
                model=model,
            )

            async def awrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                chat = await Chat.from_prompt(
                    async_prompt_function, *args, **kwargs
                ).asubmit()
                last_message = chat.messages[-1]
                while isinstance(last_message, FunctionCallMessage):
                    function_result_message = last_message.content()
                    if inspect.isawaitable(function_result_message):
                        function_result_message = await function_result_message
                    chat = chat.add_message(function_result_message).submit()
                return cast(R, chat.messages[-1].content)

            return update_wrapper(awrapper, func)

        prompt_function = PromptFunction[P, R](
            template=template,
            parameters=list(func_signature.parameters.values()),
            return_type=func_signature.return_annotation,
            functions=functions,
            model=model,
        )

        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            chat = Chat.from_prompt(prompt_function, *args, **kwargs).submit()
            while isinstance(chat.messages[-1], FunctionCallMessage):
                function_result_message = chat.messages[-1].get_result()
                chat = chat.add_message(function_result_message).submit()
            return cast(R, chat.messages[-1].content)

        return update_wrapper(wrapper, func)

    return decorator
