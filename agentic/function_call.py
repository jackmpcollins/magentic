import inspect
from typing import Any, Callable, Generic, ParamSpec, TypeVar

T = TypeVar("T")
P = ParamSpec("P")


class FunctionCall(Generic[T]):
    def __init__(self, func: Callable[P, T], *args: P.args, **kwargs: P.kwargs):
        self._func = func
        self._args = args
        self._kwargs = kwargs

    def __call__(self) -> T:
        return self._func(*self._args, **self._kwargs)

    @property
    def func(self):
        return self._func

    @property
    def arguments(self) -> dict[str, Any]:
        signature = inspect.signature(self._func)
        bound_args = signature.bind(*self._args, **self._kwargs)
        return bound_args.arguments.copy()
