# magentic

## Installation

```sh
pip install magentic
```

```sh
poetry add magentic
```

## Concepts

The `@prompt()` decorator makes it easy to convert a python function into a query to an LLM.

```python
from magentic import prompt

@prompt("What is a good name for a company that makes {product}?")
def get_company_name(product: str) -> str:
    ...  # No code required!

get_company_name(product="colorful socks")
# 'Colorful Threads'
```

The decorator will respect the return annotation of the function. This can be any builtin python type, a `pydantic` model, or a `FunctionCall`.

```python
from magentic import prompt
from pydantic import BaseModel

class Superhero(BaseModel):
    name: str
    age: int
    power: str
    enemies: list[str]

@prompt("Create a Superhero named {name}.")
def create_superhero(name: str) -> Superhero:
    ...

create_superhero("Garden Man")
# Superhero(name='Garden Man', age=30, power='Control over plants', enemies=['Pollution Man', 'Concrete Woman'])
```

A `PromptFunction` can also decide to call functions, in this case returning `FunctionCall` objects which can be called to retrieve the result.

```python
from typing import Literal

from magentic import prompt, FunctionCall


def activate_oven(temperature: int, mode: Literal["broil", "bake", "roast"]):
    """Turn the oven on with the provided settings."""
    print(f"Preheating to {temperature} F with mode {mode}")

@prompt(
    template="Prepare the oven so I can make {food}",
    functions=[activate_oven],
)
def configure_oven(food: str) -> FunctionCall[str]:
    ...


output = configure_oven("cookies!")
# output: FunctionCall[str]
output.arguments
# {'temperature': 350, 'mode': 'bake'}
output()
# 'Preheating to 350 F with mode bake'
```

To resolve `FunctionCall` objects automatically, you can use the `@prompt_chain` decorator. This will automatically resolve function calls and pass the results back to the model to continue until the final answer is reached.

```python
from magentic import prompt_chain

def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    return {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }

@prompt_chain(
    template="What's the weather like in {city}?",
    functions=[get_current_weather],
)
def describe_weather(city: str) -> str:
    ...

describe_weather("Boston")
# 'The current weather in Boston is 72°F and it is sunny and windy.'
```

## Type Checking

Many type checkers will raise warnings or errors for functions with the `prompt` decorator due to the function having no body or return value. There are several ways to deal with these.

1. Disable the check globally for the type checker. For example in mypy by disabling error code `empty-body`.
   ```toml
   # pyproject.toml
   [tool.mypy]
   disable_error_code = ["empty-body"]
   ```
1. Make the function body `...` (this does not satisfy mypy) or `raise`.
   ```python
   @prompt()
   def random_color() -> str:
       """Choose a color"""
       ...
   ```
1. Use comment `# type: ignore[empty-body]` on each function.
   ```python
   @prompt()
   def random_color() -> str:  # type: ignore[empty-body]
       """Choose a color"""
   ```
