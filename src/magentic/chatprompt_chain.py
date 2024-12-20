"""Chat prompt chain."""

import inspect
from collections.abc import Awaitable
from functools import wraps
from typing import Any, Callable, ParamSpec, TypeVar, cast

from magentic.chat import Chat
from magentic.chat_model.base import ChatModel
from magentic.chat_model.message import Message
from magentic.chatprompt import AsyncChatPromptFunction, ChatPromptDecorator, ChatPromptFunction
from magentic.function_call import (
    # AsyncParallelFunctionCall,
    FunctionCall,
    # ParallelFunctionCall,
)
from magentic.logger import logfire
from magentic.prompt_chain import MaxFunctionCallsError

P = ParamSpec("P")
R = TypeVar("R")


def chatprompt_chain(
    *messages: Message[Any],
    functions: list[Callable[..., Any]] | None = None,
    stop: list[str] | None = None,
    max_retries: int = 0,
    model: ChatModel | None = None,
    max_calls: int | None = None,
) -> ChatPromptDecorator:
    """Convert a Python function to an LLM chat prompt, auto-resolving function calls.

    The `@chatprompt_chain` decorator allows you to define a prompt template for chat-based Large Language Models (LLM).
    """

    def decorator(
        func: Callable[P, Awaitable[R]] | Callable[P, R],
    ) -> AsyncChatPromptFunction[P, R] | ChatPromptFunction[P, R]:
        func_signature = inspect.signature(func)

        if inspect.iscoroutinefunction(func):
            async_prompt_function = AsyncChatPromptFunction[P, R](
                name=func.__name__,
                parameters=list(func_signature.parameters.values()),
                # TODO: Also allow ParallelFunctionCall. Support this more neatly
                return_type=func_signature.return_annotation | FunctionCall,  # type: ignore[arg-type,unused-ignore]
                messages=messages,
                functions=functions,
                stop=stop,
                max_retries=max_retries,
                model=model,
            )

            @wraps(func)
            async def awrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
                with logfire.span(
                    f"Calling async prompt-chain {func.__name__}",
                    **func_signature.bind(*args, **kwargs).arguments,
                ):
                    chat = Chat(
                        messages=async_prompt_function.format(*args, **kwargs),
                        functions=async_prompt_function.functions,
                        output_types=async_prompt_function.return_types,
                        model=async_prompt_function.model,
                        *args,
                        **kwargs
                    ).submit()
                    num_calls = 0
                    while callable(chat.last_message.content): # was FunctionCall
                        if max_calls is not None and num_calls >= max_calls:
                            msg = (
                                f"Function {func.__name__} reached limit of"
                                f" {max_calls} function calls"
                            )
                            raise MaxFunctionCallsError(msg)
                        chat = await chat.aexec_function_call()
                        chat = await chat.asubmit()
                        num_calls += 1
                    return chat.last_message.content

            return cast(
                AsyncChatPromptFunction[P, R],
                awrapper,
            )

        prompt_function = ChatPromptFunction[P, R](
            name=func.__name__,
            parameters=list(func_signature.parameters.values()),
            # TODO: Also allow ParallelFunctionCall. Support this more neatly
            return_type=func_signature.return_annotation | FunctionCall,  # type: ignore[arg-type,unused-ignore]
            messages=messages,
            functions=functions,
            stop=stop,
            max_retries=max_retries,
            model=model,
        )

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with logfire.span(
                f"Calling prompt-chain {func.__name__}",
                **func_signature.bind(*args, **kwargs).arguments,
            ):
                chat = Chat(
                    messages=prompt_function.format(*args, **kwargs),
                    functions=prompt_function.functions,
                    output_types=prompt_function.return_types,
                    model=prompt_function.model,
                ).submit()
                num_calls = 0
                while callable(chat.last_message.content): # was FunctionCall
                    if max_calls is not None and num_calls >= max_calls:
                        msg = (
                            f"Function {func.__name__} reached limit of"
                            f" {max_calls} function calls"
                        )
                        raise MaxFunctionCallsError(msg)
                    chat = chat.exec_function_call().submit()
                    num_calls += 1
                return cast(R, chat.last_message.content)

        return cast(
            ChatPromptFunction[P, R],
            wrapper
        )

    return cast(ChatPromptDecorator, decorator)
