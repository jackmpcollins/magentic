# Overview

Easily integrate Large Language Models into your Python code. Simply use the `@prompt` and `@chatprompt` decorators to create functions that return structured output from the LLM. Mix LLM queries and function calling with regular Python code to create complex logic.

## Features

- [Structured Outputs](structured-outputs.md) using pydantic models and built-in python types.
- [Chat Prompting](chat-prompting.md) to enable few-shot prompting with structured examples.
- [Function Calling](function-calling.md) and [Parallel Function Calling](function-calling.md) via the `FunctionCall` and `ParallelFunctionCall` return types.
- [Formatting](formatting.md) to naturally insert python objects into prompts.
- [Asyncio](asyncio.md). Simply use `async def` when defining a magentic function.
- [Streaming](streaming.md) structured outputs to use them as they are being generated.
- [Vision](vision.md) to easily get stuctured outputs from images.
- Multiple LLM providers including OpenAI and Anthropic. See [Configuration](configuration.md).
- [Type Checking](type-checking.md) to work nicely with linters and IDEs.

## Installation

```sh
pip install magentic
```

or using poetry

```sh
poetry add magentic
```

Configure your OpenAI API key by setting the `OPENAI_API_KEY` environment variable. To configure a different LLM provider see [Configuration](configuration.md) for more.

## Usage

### @prompt

The `@prompt` decorator allows you to define a template for a Large Language Model (LLM) prompt as a Python function. When this function is called, the arguments are inserted into the template, then this prompt is sent to an LLM which generates the function output.

```python
from magentic import prompt


@prompt('Add more "dude"ness to: {phrase}')
def dudeify(phrase: str) -> str: ...  # No function body as this is never executed


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
def create_superhero(name: str) -> Superhero: ...


create_superhero("Garden Man")
# Superhero(name='Garden Man', age=30, power='Control over plants', enemies=['Pollution Man', 'Concrete Woman'])
```

See [Structured Outputs](structured-outputs.md) for more.

### @chatprompt

The `@chatprompt` decorator works just like `@prompt` but allows you to pass chat messages as a template rather than a single text prompt. This can be used to provide a system message or for few-shot prompting where you provide example responses to guide the model's output. Format fields denoted by curly braces `{example}` will be filled in all messages (except `FunctionResultMessage`).

```python
from magentic import chatprompt, AssistantMessage, SystemMessage, UserMessage
from pydantic import BaseModel


class Quote(BaseModel):
    quote: str
    character: str


@chatprompt(
    SystemMessage("You are a movie buff."),
    UserMessage("What is your favorite quote from Harry Potter?"),
    AssistantMessage(
        Quote(
            quote="It does not do to dwell on dreams and forget to live.",
            character="Albus Dumbledore",
        )
    ),
    UserMessage("What is your favorite quote from {movie}?"),
)
def get_movie_quote(movie: str) -> Quote: ...


get_movie_quote("Iron Man")
# Quote(quote='I am Iron Man.', character='Tony Stark')
```

See [Chat Prompting](chat-prompting.md) for more.

### FunctionCall

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
def configure_oven(food: str) -> FunctionCall[str]: ...


output = configure_oven("cookies!")
# FunctionCall(<function activate_oven at 0x1105a6200>, temperature=350, mode='bake')
output()
# 'Preheating to 350 F with mode bake'
```

See [Function Calling](function-calling.md) for more.

### @prompt_chain

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
def describe_weather(city: str) -> str: ...


describe_weather("Boston")
# 'The current weather in Boston is 72°F and it is sunny and windy.'
```

LLM-powered functions created using `@prompt`, `@chatprompt` and `@prompt_chain` can be supplied as `functions` to other `@prompt`/`@prompt_chain` decorators, just like regular python functions. This enables increasingly complex LLM-powered functionality, while allowing individual components to be tested and improved in isolation.
