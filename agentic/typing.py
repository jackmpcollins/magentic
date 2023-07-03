import types
from typing import Sequence, TypeGuard, TypeVar, Union, get_args, get_origin


def is_union_type(type_: type) -> bool:
    type_ = get_origin(type_) or type_
    return type_ is Union or type_ is types.UnionType  # noqa: E721


T = TypeVar("T")


def split_union_type(type_: type[T]) -> Sequence[type[T]]:
    return get_args(type_) if is_union_type(type_) else [type_]


TypeT = TypeVar("TypeT", bound=type)


def is_origin_subclass(
    type_: type, cls_or_tuple: TypeT | tuple[TypeT, ...]
) -> TypeGuard[TypeT]:
    """Check if the unsubscripted type is a subclass of the given class(es)."""
    return issubclass(get_origin(type_) or type_, cls_or_tuple)
