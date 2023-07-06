import inspect
from functools import update_wrapper
from typing import Any, Callable, Generic, ParamSpec, Sequence, TypeVar

from agentic.function_call import FunctionCall
from agentic.openai_chat_model import OpenaiChatModel, UserMessage
from agentic.typing import is_origin_subclass, split_union_type

P = ParamSpec("P")
# TODO: Make `R` type Union of all possible return types except FunctionCall ?
# Then `R | FunctionCall[FuncR]` will separate FunctionCall from other return types.
# Can then use `FuncR` to make typechecker check `functions` argument to `prompt`
# `Not` type would solve this - https://github.com/python/typing/issues/801
R = TypeVar("R")


class PromptFunction(Generic[P, R]):
    def __init__(
        self,
        template: str,
        parameters: Sequence[inspect.Parameter],
        return_type: type[R],
        functions: list[Callable[..., Any]] | None = None,
    ):
        self._signature = inspect.Signature(
            parameters=parameters,
            return_annotation=return_type,
        )
        self._template = template
        self._functions = functions or []

        self._return_types = [
            return_type
            for return_type in split_union_type(return_type)
            if not is_origin_subclass(return_type, FunctionCall)
        ]

        self._model = OpenaiChatModel()

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        bound_args = self._signature.bind(*args, **kwargs)
        bound_args.apply_defaults()
        return self._model.complete(  # type: ignore[return-value]
            messages=[
                UserMessage(content=self._template.format(**bound_args.arguments))
            ],
            functions=self._functions,
            output_types=self._return_types,
        )

    @property
    def functions(self) -> list[Callable[..., Any]]:
        return self._functions.copy()

    def format(self, *args: P.args, **kwargs: P.kwargs) -> str:
        bound_args = self._signature.bind(*args, **kwargs)
        bound_args.apply_defaults()
        return self._template.format(**bound_args.arguments)


def prompt(
    functions: list[Callable[..., Any]] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        if func.__doc__ is None:
            raise ValueError("Function must have a docstring")

        func_signature = inspect.Signature.from_callable(func)
        prompt_function = PromptFunction[P, R](
            parameters=list(func_signature.parameters.values()),
            return_type=func_signature.return_annotation,
            template=func.__doc__,
            functions=functions,
        )
        return update_wrapper(prompt_function, func)

    return decorator
