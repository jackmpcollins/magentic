import inspect
from collections.abc import Callable, Sequence
from functools import wraps
from typing import Any, ParamSpec, TypeVar, cast

from magentic.streaming import StreamedStr
from magentic import ParallelFunctionCall
from magentic._streamed_response import StreamedResponse, AsyncStreamedResponse
from magentic._chat import Chat
from magentic.chat_model.base import ChatModel
from magentic.chat_model.message import Message, UserMessage
from magentic.chatprompt import AsyncChatPromptFunction, ChatPromptFunction
from magentic.function_call import FunctionCall, AsyncParallelFunctionCall
from magentic.logger import logfire

P = ParamSpec("P")
R = TypeVar("R")


class MaxFunctionCallsError(Exception):
    """Raised when prompt chain reaches the max number of function calls."""


async def list_to_async_iter(list):
    for item in list:
        yield item


def prompt_chain(
    template: str | Sequence[Message[Any]],
    functions: list[Callable[..., Any]] | None = None,
    model: ChatModel | None = None,
    max_calls: int | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    messages = (
        [UserMessage(content=template)] if isinstance(template, str) else template
    )

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        func_signature = inspect.signature(func)
        if inspect.iscoroutinefunction(func):
            async_prompt_function = AsyncChatPromptFunction[P, Any](
                name=func.__name__,
                parameters=list(func_signature.parameters.values()),
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
                    while isinstance(chat.last_message.content, AsyncStreamedResponse):
                        function_calls = []
                        is_break = True
                        async for item in chat.last_message.content:
                            if isinstance(item, FunctionCall):
                                function_calls.append(item)
                                is_break = False
                        if max_calls is not None and num_calls >= max_calls:
                            msg = (
                                f"Function {func.__name__} reached limit of"
                                f" {max_calls} function calls"
                            )
                            raise MaxFunctionCallsError(msg)
                        if len(function_calls) == 1:
                            chat = await chat.aexec_function_call(function_calls[0])
                            chat = await chat.asubmit()
                            num_calls += 1
                        elif len(function_calls) > 1:
                            function_calls = list_to_async_iter(function_calls)
                            multi_functions = AsyncParallelFunctionCall(function_calls)
                            chat = await chat.aexec_function_call(multi_functions)
                            chat = await chat.asubmit()
                            num_calls += len(function_calls)
                        if is_break:
                            break
                    return chat.last_message.content, chat.messages

            return cast(Callable[P, R], awrapper)
        prompt_function = ChatPromptFunction[P, R](
            name=func.__name__,
            parameters=list(func_signature.parameters.values()),
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
                while isinstance(chat.last_message.content, StreamedResponse):
                    function_calls = []
                    is_break = True
                    for item in chat.last_message.content:
                        if isinstance(item, FunctionCall):
                            function_calls.append(item)
                            is_break = False
                    if max_calls is not None and num_calls >= max_calls:
                        msg = (
                            f"Function {func.__name__} reached limit of"
                            f" {max_calls} function calls"
                        )
                        raise MaxFunctionCallsError(msg)

                    if len(function_calls) == 1:
                        chat = chat.exec_function_call(function_calls[0]).submit()
                        num_calls += 1
                    elif len(function_calls) > 1:
                        multi_functions = ParallelFunctionCall(function_calls)
                        chat = chat.exec_function_call(multi_functions).submit()
                        num_calls += len(function_calls)
                    if is_break:
                        break
                return chat.last_message.content, chat.messages

        return wrapper

    return decorator
