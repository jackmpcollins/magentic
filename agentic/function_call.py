from typing import Generic, ParamSpec, TypeVar, Callable

T = TypeVar("T")
P = ParamSpec("P")


class FunctionCall(Generic[T]):
    def __init__(self, func: Callable[P, T], *args: P.args, **kwargs: P.kwargs):
        self._func = func
        self._args = args
        self._kwargs = kwargs

    def __call__(self) -> T:
        return self._func(*self._args, **self._kwargs)
