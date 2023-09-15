import inspect
from typing import Any, Callable, Generic, ParamSpec, TypeVar

T = TypeVar("T")
P = ParamSpec("P")


class FunctionCall(Generic[T]):
    """A function with arguments supplied.

    Calling the instance will call the function with the supplied arguments.
    """

    def __init__(self, function: Callable[P, T], *args: P.args, **kwargs: P.kwargs):
        self._function = function
        self._args = args
        self._kwargs = kwargs

    def __call__(self) -> T:
        return self._function(*self._args, **self._kwargs)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return (
            type(self) is type(other)
            and self._function == other._function
            and self._args == other._args
            and self._kwargs == other._kwargs
        )

    def __repr__(self) -> str:
        args_kwargs_repr = ", ".join(
            [
                *(repr(arg) for arg in self._args),
                *(f"{key}={value!r}" for key, value in self._kwargs.items()),
            ]
        )
        return f"{type(self).__name__}({self._function!r}, {args_kwargs_repr})"

    @property
    def function(self):
        return self._function

    @property
    def arguments(self) -> dict[str, Any]:
        signature = inspect.signature(self._function)
        bound_args = signature.bind(*self._args, **self._kwargs)
        return bound_args.arguments.copy()
