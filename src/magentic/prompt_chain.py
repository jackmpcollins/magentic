import inspect
from collections.abc import Callable, Sequence
from functools import wraps
from typing import Any, ParamSpec, TypeVar, cast

from magentic._chat import Chat
from magentic.chat_model.base import ChatModel
from magentic.chat_model.message import Message, UserMessage
from magentic.chatprompt import AsyncChatPromptFunction, ChatPromptFunction
from magentic.function_call import FunctionCall
from magentic.logger import logfire

P = ParamSpec("P")
R = TypeVar("R")


class MaxFunctionCallsError(Exception):
    """Raised when prompt chain reaches the max number of function calls."""


def prompt_chain(
    template: str | Sequence[Message[Any]],
    functions: list[Callable[..., Any]] | None = None,
    model: ChatModel | None = None,
    max_calls: int | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Convert a Python function to an LLM query, auto-resolving function calls.

    Use `@prompt_chain` when you need the LLM to perform multiple function calls to
    reach a final answer. When a function decorated with `@prompt_chain` is called, the
    LLM is queried, then any function calls are automatically executed and the results
    appended to the list of messages. Then the LLM is queried again and this repeats
    until a final answer is reached.

    Set `max_calls` to limit the number of function calls. If the limit is reached, a
    `MaxFunctionCallsError` will be raised.
    """

    messages = (
        [UserMessage(content=template)] if isinstance(template, str) else template
    )

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        func_signature = inspect.signature(func)

        if inspect.iscoroutinefunction(func):
            async_prompt_function = AsyncChatPromptFunction[P, Any](
                name=func.__name__,
                parameters=list(func_signature.parameters.values()),
                # TODO: Also allow ParallelFunctionCall. Support this more neatly
                return_type=func_signature.return_annotation | FunctionCall,  # type: ignore[arg-type,unused-ignore]
                messages=messages,
                functions=functions,
                model=model,
            )

            @wraps(func)
            async def awrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
                with logfire.span(
                    f"Calling async prompt-chain {func.__name__}",
                    **func_signature.bind(*args, **kwargs).arguments,
                ):
                    chat = await Chat(
                        messages=async_prompt_function.format(*args, **kwargs),
                        functions=async_prompt_function.functions,
                        output_types=async_prompt_function.return_types,
                        model=async_prompt_function._model,  # Keep `None` value if unset
                    ).asubmit()
                    num_calls = 0
                    while isinstance(chat.last_message.content, FunctionCall):
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

            return cast(Callable[P, R], awrapper)

        prompt_function = ChatPromptFunction[P, R](
            name=func.__name__,
            parameters=list(func_signature.parameters.values()),
            # TODO: Also allow ParallelFunctionCall. Support this more neatly
            return_type=func_signature.return_annotation | FunctionCall,  # type: ignore[arg-type,unused-ignore]
            messages=messages,
            functions=functions,
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
                    model=prompt_function._model,  # Keep `None` value if unset
                ).submit()
                num_calls = 0
                while isinstance(chat.last_message.content, FunctionCall):
                    if max_calls is not None and num_calls >= max_calls:
                        msg = (
                            f"Function {func.__name__} reached limit of"
                            f" {max_calls} function calls"
                        )
                        raise MaxFunctionCallsError(msg)
                    chat = chat.exec_function_call().submit()
                    num_calls += 1
                return cast(R, chat.last_message.content)

        return wrapper

    return decorator
