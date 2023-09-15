import inspect
from functools import wraps
from typing import (
    Any,
    Callable,
    ParamSpec,
    TypeVar,
    cast,
)

from magentic.chat import Chat
from magentic.chat_model.openai_chat_model import OpenaiChatModel
from magentic.function_call import FunctionCall
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
            async_prompt_function = AsyncPromptFunction[P, Any](
                template=template,
                parameters=list(func_signature.parameters.values()),
                return_type=func_signature.return_annotation,
                functions=functions,
                model=model,
            )

            @wraps(func)
            async def awrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
                chat = await Chat.from_prompt(
                    async_prompt_function, *args, **kwargs
                ).asubmit()
                while isinstance(chat.messages[-1].content, FunctionCall):
                    chat = await chat.aexec_function_call()
                    chat = await chat.asubmit()
                return chat.messages[-1].content

            return cast(Callable[P, R], awrapper)

        prompt_function = PromptFunction[P, R](
            template=template,
            parameters=list(func_signature.parameters.values()),
            return_type=func_signature.return_annotation,
            functions=functions,
            model=model,
        )

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            chat = Chat.from_prompt(prompt_function, *args, **kwargs).submit()
            while isinstance(chat.messages[-1].content, FunctionCall):
                chat = chat.exec_function_call().submit()
            return cast(R, chat.messages[-1].content)

        return wrapper

    return decorator
