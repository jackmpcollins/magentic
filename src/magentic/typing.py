import inspect
import types
from typing import (
    Any,
    Mapping,
    Sequence,
    TypeGuard,
    TypeVar,
    Union,
    get_args,
    get_origin,
)


def is_union_type(type_: type) -> bool:
    """Return True if the type is a union type."""
    type_ = get_origin(type_) or type_
    return type_ is Union or type_ is types.UnionType  # noqa: E721


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
    if type_ is Any:
        return False
    return issubclass(get_origin(type_) or type_, cls_or_tuple)


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

    if name := getattr(type_, "__name__", None):
        assert isinstance(name, str)

        if len(args) == 1:
            return f"{name.lower()}_of_{name_type(args[0])}"

        return name.lower()

    raise ValueError(f"Unable to name type {type_}")
