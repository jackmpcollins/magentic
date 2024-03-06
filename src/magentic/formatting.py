from collections import OrderedDict
from typing import TypeVar

T = TypeVar("T")


class BulletedList(list[T]):
    """A list of items that is formatted as a bulleted list.

    When an instance of this class is formatted using the `format` function
    or the `f-string` syntax, it will be formatted as a bulleted list.

    Example:
    --------
    Create an instance of BulletedList and insert it into an f-string:

        items = BulletedList(["foo", "bar", "baz"])
        print(f"{items}")

    This will output:

        - foo
        - bar
        - baz
    """

    def __format__(self, format_spec: str) -> str:
        return "\n".join(f"- {item}" for item in self)

    def __repr__(self) -> str:
        return f"BulletedList({super().__repr__()})"


class NumberedList(list[T]):
    """A list of items that is formatted as a numbered list.

    When an instance of this class is formatted using the `format` function
    or the `f-string` syntax, it will be formatted as a numbered list.

    Example:
    --------
    Create an instance of NumberedList and insert it into an f-string:

        items = NumberedList(["foo", "bar", "baz"])
        print(f"{items}")

    This will output:

        1. foo
        2. bar
        3. baz
    """

    def __format__(self, format_spec: str) -> str:
        return "\n".join(f"{i}. {item}" for i, item in enumerate(self, 1))

    def __repr__(self) -> str:
        return f"NumberedList({super().__repr__()})"


K = TypeVar("K")
V = TypeVar("V")


class BulletedDict(OrderedDict[K, V]):
    """A dictionary that is formatted as a bulleted list.

    When an instance of this class is formatted using the `format` function
    or the `f-string` syntax, it will be formatted as a bulleted list.

    Example
    -------
    Create an instance of BulletedDict and insert it into an f-string:

        items = BulletedDict({"foo": 1, "bar": 2, "baz": 3})
        print(f"{items}")

    This will output:

        - foo: 1
        - bar: 2
        - baz: 3
    """

    def __format__(self, format_spec: str) -> str:
        return "\n".join(f"- {key}: {value}" for key, value in self.items())

    def __repr__(self) -> str:
        return f"BulletedDict({super().__repr__()})"


class NumberedDict(OrderedDict[K, V]):
    """A dictionary that is formatted as a numbered list.

    When an instance of this class is formatted using the `format` function
    or the `f-string` syntax, it will be formatted as a numbered list.

    Example
    -------
    Create an instance of NumberedDict and insert it into an f-string:

        items = NumberedDict({"foo": 1, "bar": 2, "baz": 3})
        print(f"{items}")

    This will output:

        1. foo: 1
        2. bar: 2
        3. baz: 3
    """

    def __format__(self, format_spec: str) -> str:
        return "\n".join(
            f"{i}. {key}: {value}" for i, (key, value) in enumerate(self.items(), 1)
        )

    def __repr__(self) -> str:
        return f"NumberedDict({super().__repr__()})"
