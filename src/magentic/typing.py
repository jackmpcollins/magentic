import inspect
import types
from collections.abc import Iterable, Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    TypeGuard,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

T_co = TypeVar("T_co", covariant=True)

if TYPE_CHECKING:
    # Cannot be defined at runtime because Protocol cannot inherit from non-Protocol
    class NonStringSequence(Sequence[T_co], Protocol[T_co]):  # type: ignore[misc]
        """Protocol that matches Sequences except for `str`."""

        # HACK: Works because `__contains__` method of `str` does not match `Sequence`
        # See: https://github.com/python/typing/issues/256#issuecomment-1442633430


def is_union_type(type_: type) -> bool:
    """Return True if the type is a union type."""
    type_ = get_origin(type_) or type_
    return type_ is Union or type_ is types.UnionType


TypeT = TypeVar("TypeT", bound=type)


def split_union_type(type_: TypeT) -> Sequence[TypeT]:
    """Split a union type into its constituent types."""
    return get_args(type_) if is_union_type(type_) else [type_]


def is_origin_abstract(type_: type) -> bool:
    """Return true if the unsubscripted type is an abstract base class (ABC)."""
    return inspect.isabstract(get_origin(type_) or type_)


def is_origin_subclass(
    type_: type, cls_or_tuple: TypeT | tuple[TypeT, ...]
) -> TypeGuard[TypeT]:
    """Check if the unsubscripted type is a subclass of the given class(es)."""
    if type_ is Any:  # type: ignore[comparison-overlap]
        return False
    return issubclass(get_origin(type_) or type_, cls_or_tuple)


def is_any_origin_subclass(
    types: Iterable[type], cls_or_tuple: TypeT | tuple[TypeT, ...]
) -> bool:
    """Check if any of the unsubscripted types is a subclass of the given class(es)."""
    return any(is_origin_subclass(type_, cls_or_tuple) for type_ in types)


def name_type(type_: type) -> str:
    """Generate a name for the given type.

    e.g. `list[str]` -> `"list_of_str"`
    """
    if is_origin_subclass(type_, types.NoneType):
        return "null"

    if is_union_type(type_):
        return "_or_".join(name_type(arg) for arg in split_union_type(type_))

    args = get_args(type_)

    if is_origin_subclass(type_, Mapping) and len(args) == 2:
        key_type, value_type = args
        return f"dict_of_{name_type(key_type)}_to_{name_type(value_type)}"

    pydantic_metadata = getattr(type_, "__pydantic_generic_metadata__", None)
    if (
        pydantic_metadata
        and (origin := pydantic_metadata.get("origin"))
        and (args := pydantic_metadata.get("args"))
    ):
        return name_type(origin) + "_" + "_".join(name_type(arg) for arg in args)

    if name := getattr(type_, "__name__", None):
        assert isinstance(name, str)

        if len(args) == 1:
            return f"{name.lower()}_of_{name_type(args[0])}"

        return name.lower()

    msg = f"Unable to name type {type_}"
    raise ValueError(msg)
