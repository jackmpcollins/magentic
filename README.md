# magentic

## Installation

```sh
pip install magentic
```

or using poetry

```sh
poetry add magentic
```

Configure your OpenAI API key by setting the `OPENAI_API_KEY` environment variable or using `openai.api_key = "sk-..."`. See the [OpenAI Python library documentation](https://github.com/openai/openai-python#usage) for more information.

## Concepts

The `@prompt` decorator allows you to define a template prompt for a Large Language Model (LLM) as a Python function.

```python
from magentic import prompt


@prompt("What is a good name for a company that makes {product}?")
def get_company_name(product: str) -> str:
    ...  # No code required!


get_company_name(product="colorful socks")
# 'Colorful Threads'
```

The decorator will respect the return type annotation of the function. This can be [any type supported by `pydantic`](https://docs.pydantic.dev/latest/usage/types/types/) including a `pydantic` model.

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

An LLM can also decide to call functions. In this case the `@prompt`-decorated function returns a `FunctionCall` object which can be called to execute the function using the arguments provided by the LLM.

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
# FunctionCall(<function activate_oven at 0x1105a6200>, temperature=350, mode='bake')
output()
# 'Preheating to 350 F with mode bake'
```

Sometimes the LLM requires making one or more function calls to generate a final answer. The `@prompt_chain` decorator will resolve `FunctionCall` objects automatically and pass the results back to the LLM to continue until the final answer is reached. In the following example, the LLM calls the `get_current_weather` function and then uses the result to formulate its final answer.

```python
from magentic import prompt_chain


def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    # Pretend to query an API
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

### Additional Features

- The docstring of the decorated function will be used as the prompt template if the `template` argument is not provided to `@prompt`/`@prompt_chain`.
- The `Annotated` type annotation can be used to provide descriptions and other metadata for function parameters. See [the `pydantic` documentation on using `Field` to describe function arguments](https://docs.pydantic.dev/latest/usage/validation_decorator/#using-field-to-describe-function-arguments).
- The `@prompt` and `@prompt_chain` decorators also accept a `model` argument. You can pass an instance of `OpenaiChatModel` (from `magentic.chat_model.openai_chat_model`) to use GPT4 or configure a different temperature.

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
