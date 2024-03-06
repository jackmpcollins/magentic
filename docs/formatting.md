# Formatting

## The `format` Method

Functions created using magentic decorators expose a `format` method that accepts the same parameters as the function itself but returns the completed prompt that will be sent to the model. For `@prompt` this method returns a string, and for `@chatprompt` it returns a list of `Message` objects. The `format` method can be used to test that the final prompt created by a magentic function is formatted as expected.

```python
from magentic import prompt


@prompt("Write a short poem about {topic}.")
def create_poem(topic: str) -> str: ...


create_poem.format("fruit")
# 'Write a short poem about fruit.'
```

## Classes for Formatting

By default, when a list is used in a prompt template string it is formatted using its Python representation.

```python
from magentic import prompt
from magentic.formatting import BulletedList


@prompt("Continue the list:\n{items}")
def get_next_items(items: list[str]) -> list[str]: ...


items = ["apple", "banana", "cherry"]
print(get_next_items.format(items=items))
# Continue the list:
# ['apple', 'banana', 'cherry']
```

However, the LLM might respond better to a prompt in which the list is formatted more clearly or the items are numbered. The `BulletedList`, `NumberedList`, `BulletedDict` and `NumberedDict` classes are provided to enable this.

For example, to modify the above prompt to contain a numbered list of the items, the `NumberedList` class can be used. This behaves exactly like a regular Python `list` except for how it appears when inserted into a formatted string. This class can also be used as the type annotation for `items` parameter to ensure that this prompt always contains a numbered list.

```python hl_lines="6 10"
from magentic import prompt
from magentic.formatting import NumberedList


@prompt("Continue the list:\n{items}")
def get_next_items(items: NumberedList[str]) -> list[str]: ...


items = NumberedList(["apple", "banana", "cherry"])
print(get_next_items.format(items=items))
# Continue the list:
# 1. apple
# 2. banana
# 3. cherry
```

## Custom Formatting

When objects are inserted into formatted strings in Python, the `__format__` method is called. By defining or modifying this method you can control how an object is converted to a string in the prompt. If you own the class you can modify the `__format__` method directly. Otherwise for third-party classes you will need to create a subcless.

Here's an example of how to represent a dictionary as a bulleted list.

```python
from typing import TypeVar

from magentic import prompt

K = TypeVar("K")
V = TypeVar("V")


class BulletedDict(dict[K, V]):
    def __format__(self, format_spec: str) -> str:
        # Here, you could use 'format_spec' to customize the formatting further if needed
        return "\n".join(f"- {key}: {value}" for key, value in self.items())


@prompt("Identify the odd one out:\n{items}")
def find_odd_one_out(items: BulletedDict[str, str]) -> str: ...


items = BulletedDict({"sky": "blue", "grass": "green", "sun": "purple"})
print(find_odd_one_out.format(items))
# Identify the odd one out:
# - sky: blue
# - grass: green
# - sun: purple
```
