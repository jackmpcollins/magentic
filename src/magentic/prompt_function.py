import copy
import inspect
from functools import update_wrapper
from typing import (
    Any,
    Awaitable,
    Callable,
    Generic,
    ParamSpec,
    Protocol,
    Sequence,
    TypeVar,
    cast,
    overload,
)

from magentic.backend import get_chat_model
from magentic.chat_model.base import ChatModel
from magentic.chat_model.message import UserMessage
from magentic.function_call import FunctionCall
from magentic.typing import is_origin_subclass, split_union_type

P = ParamSpec("P")
# TODO: Make `R` type Union of all possible return types except FunctionCall ?
# Then `R | FunctionCall[FuncR]` will separate FunctionCall from other return types.
# Can then use `FuncR` to make typechecker check `functions` argument to `prompt`
# `Not` type would solve this - https://github.com/python/typing/issues/801
R = TypeVar("R")


class BasePromptFunction(Generic[P, R]):
    """Base class for an LLM prompt template that is directly callable to query the LLM."""

    def __init__(
        self,
        template: str,
        parameters: Sequence[inspect.Parameter],
        return_type: type[R],
        functions: list[Callable[..., Any]] | None = None,
        stop: list[str] | None = None,
        model: ChatModel | None = None,
    ):
        self._signature = inspect.Signature(
            parameters=parameters,
            return_annotation=return_type,
        )
        self._template = template
        self._functions = functions or []
        self._stop = stop
        self._model = model

        self._return_types = [
            type_
            for type_ in split_union_type(return_type)
            if not is_origin_subclass(type_, FunctionCall)
        ]

    @property
    def functions(self) -> list[Callable[..., Any]]:
        return self._functions.copy()

    @property
    def stop(self) -> list[str] | None:
        return copy.copy(self._stop)

    @property
    def model(self) -> ChatModel:
        return self._model or get_chat_model()

    @property
    def return_types(self) -> list[type[R]]:
        return self._return_types.copy()

    def format(self, *args: P.args, **kwargs: P.kwargs) -> str:
        """Format the prompt template with the given arguments."""
        bound_args = self._signature.bind(*args, **kwargs)
        bound_args.apply_defaults()
        return self._template.format(**bound_args.arguments)


class PromptFunction(BasePromptFunction[P, R], Generic[P, R]):
    """An LLM prompt template that is directly callable to query the LLM."""

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Query the LLM with the formatted prompt template."""
        message = self.model.complete(
            messages=[UserMessage(content=self.format(*args, **kwargs))],
            functions=self._functions,
            output_types=self._return_types,
            stop=self._stop,
        )
        return cast(R, message.content)


class AsyncPromptFunction(BasePromptFunction[P, R], Generic[P, R]):
    """Async version of `PromptFunction`."""

    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Asynchronously query the LLM with the formatted prompt template."""
        message = await self.model.acomplete(
            messages=[UserMessage(content=self.format(*args, **kwargs))],
            functions=self._functions,
            output_types=self._return_types,
            stop=self._stop,
        )
        return cast(R, message.content)


class PromptDecorator(Protocol):
    """Protocol for a decorator that returns a `PromptFunction`.

    This allows finer-grain type annotation of the `prompt` function
    See https://github.com/microsoft/pyright/issues/5014#issuecomment-1523778421
    """

    @overload
    def __call__(self, func: Callable[P, Awaitable[R]]) -> AsyncPromptFunction[P, R]:  # type: ignore[overload-overlap]
        ...

    @overload
    def __call__(self, func: Callable[P, R]) -> PromptFunction[P, R]:
        ...


def prompt(
    template: str,
    functions: list[Callable[..., Any]] | None = None,
    stop: list[str] | None = None,
    model: ChatModel | None = None,
) -> PromptDecorator:
    """Convert a function into an LLM prompt template.

    The `@prompt` decorator allows you to define a template for a Large Language Model (LLM) prompt as a function.
    When this function is called, the arguments are inserted into the template, then this prompt is sent to an LLM which
    generates the function output.

    Examples
    --------
    >>> @prompt("Add more dudeness to: {phrase}")
    >>> def dudeify(phrase: str) -> str:
    >>>     ...  # No function body as this is never executed
    >>>
    >>> dudeify("Hello, how are you?")
    "Hey, dude! What's up? How's it going, my man?"
    """

    def decorator(
        func: Callable[P, Awaitable[R]] | Callable[P, R],
    ) -> AsyncPromptFunction[P, R] | PromptFunction[P, R]:
        func_signature = inspect.signature(func)

        if inspect.iscoroutinefunction(func):
            async_prompt_function = AsyncPromptFunction[P, R](
                template=template,
                parameters=list(func_signature.parameters.values()),
                return_type=func_signature.return_annotation,
                functions=functions,
                stop=stop,
                model=model,
            )
            async_prompt_function = update_wrapper(async_prompt_function, func)
            return cast(AsyncPromptFunction[P, R], async_prompt_function)  # type: ignore[redundant-cast]

        prompt_function = PromptFunction[P, R](
            template=template,
            parameters=list(func_signature.parameters.values()),
            return_type=func_signature.return_annotation,
            functions=functions,
            stop=stop,
            model=model,
        )
        return cast(PromptFunction[P, R], update_wrapper(prompt_function, func))  # type: ignore[redundant-cast]

    return cast(PromptDecorator, decorator)
