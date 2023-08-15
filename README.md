# magentic

Easily integrate Large Language Models into your Python code. Simply use the `@prompt` decorator to create functions that return structured output from the LLM. Mix LLM queries and function calling with regular Python code to create complex logic.

`magentic` is

- **Compact:** Query LLMs without duplicating boilerplate code.
- **Atomic:** Prompts are functions that can be individually tested and reasoned about.
- **Transparent:** Create "chains" using regular Python code. Define all of your own prompts.
- **Compatible:** Use `@prompt` functions as normal functions, including with decorators like `@lru_cache`.
- **Type Annotated:** Works with linters and IDEs.

Continue reading for sample usage, or go straight to the [examples directory](examples/).

## Installation

```sh
pip install magentic
```

or using poetry

```sh
poetry add magentic
```

Configure your OpenAI API key by setting the `OPENAI_API_KEY` environment variable or using `openai.api_key = "sk-..."`. See the [OpenAI Python library documentation](https://github.com/openai/openai-python#usage) for more information.

## Usage

The `@prompt` decorator allows you to define a template for a Large Language Model (LLM) prompt as a Python function. When this function is called, the arguments are inserted into the template, then this prompt is sent to an LLM which generates the function output.

```python
from magentic import prompt


@prompt('Add more "dude"ness to: {phrase}')
def dudeify(phrase: str) -> str:
    ...  # No function body as this is never executed


dudeify("Hello, how are you?")
# "Hey, dude! What's up? How's it going, my man?"
```

The `@prompt` decorator will respect the return type annotation of the decorated function. This can be [any type supported by pydantic](https://docs.pydantic.dev/latest/usage/types/types/) including a `pydantic` model.

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


def activate_oven(temperature: int, mode: Literal["broil", "bake", "roast"]) -> str:
    """Turn the oven on with the provided settings."""
    return f"Preheating to {temperature} F with mode {mode}"


@prompt(
    "Prepare the oven so I can make {food}",
    functions=[activate_oven],
)
def configure_oven(food: str) -> FunctionCall[str]:
    ...


output = configure_oven("cookies!")
# FunctionCall(<function activate_oven at 0x1105a6200>, temperature=350, mode='bake')
output()
# 'Preheating to 350 F with mode bake'
```

Sometimes the LLM requires making one or more function calls to generate a final answer. The `@prompt_chain` decorator will resolve `FunctionCall` objects automatically and pass the output back to the LLM to continue until the final answer is reached.

In the following example, when `describe_weather` is called the LLM first calls the `get_current_weather` function, then uses the result of this to formulate its final answer which gets returned.

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
    "What's the weather like in {city}?",
    functions=[get_current_weather],
)
def describe_weather(city: str) -> str:
    ...


describe_weather("Boston")
# 'The current weather in Boston is 72°F and it is sunny and windy.'
```

LLM-powered functions created using `@prompt` and `@prompt_chain` can be supplied as `functions` to other `@prompt`/`@prompt_chain` decorators, just like regular python functions. This enables increasingly complex LLM-powered functionality, while allowing individual components to be tested and improved in isolation.

See the [examples directory](examples/) for more.

### Streaming

The `StreamedStr` (and `AsyncStreamedStr`) class can be used to stream the output of the LLM. This allows you to process the text while it is being generated, rather than receiving the whole output at once. Multiple `StreamedStr` can be created at the same time to stream LLM outputs concurrently. In the below example, generating the description for multiple countries takes approximately the same amount of time as for a single country.

```python
from magentic import prompt, StreamedStr


@prompt("Tell me about {country}")
def describe_country(country: str) -> StreamedStr:
    ...


# Print the chunks while they are being received
for chunk in describe_country("Brazil"):
    print(chunk, end="")
# 'Brazil, officially known as the Federative Republic of Brazil, is ...'


# Generate text concurrently by creating the streams before consuming them
streamed_strs = [describe_country(c) for c in ["Australia", "Brazil", "Chile"]]
[str(s) for s in streamed_strs]
# ["Australia is a country ...", "Brazil, officially known as ...", "Chile, officially known as ..."]
```

### Additional Features

- The `@prompt` decorator can also be used with `async` function definitions, which enables making concurrent queries to the LLM.
- The `Annotated` type annotation can be used to provide descriptions and other metadata for function parameters. See [the pydantic documentation on using `Field` to describe function arguments](https://docs.pydantic.dev/latest/usage/validation_decorator/#using-field-to-describe-function-arguments).
- The `@prompt` and `@prompt_chain` decorators also accept a `model` argument. You can pass an instance of `OpenaiChatModel` (from `magentic.chat_model.openai_chat_model`) to use GPT4 or configure a different temperature.

## Type Checking

Many type checkers will raise warnings or errors for functions with the `@prompt` decorator due to the function having no body or return value. There are several ways to deal with these.

1. Disable the check globally for the type checker. For example in mypy by disabling error code `empty-body`.
   ```toml
   # pyproject.toml
   [tool.mypy]
   disable_error_code = ["empty-body"]
   ```
1. Make the function body `...` (this does not satisfy mypy) or `raise`.
   ```python
   @prompt("Choose a color")
   def random_color() -> str:
       ...
   ```
1. Use comment `# type: ignore[empty-body]` on each function. In this case you can add a docstring instead of `...`.
   ```python
   @prompt("Choose a color")
   def random_color() -> str:  # type: ignore[empty-body]
       """Returns a random color."""
   ```
