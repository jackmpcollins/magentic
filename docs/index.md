# Overview

Seamlessly integrate Large Language Models into Python code. Use the `@prompt` and `@chatprompt` decorators to create functions that return structured output from an LLM. Combine LLM queries and tool use with traditional Python code to build complex agentic systems.

## Features

- [Structured Outputs] using pydantic models and built-in python types.
- [Streaming] of structured outputs and function calls, to use them while being generated.
- [LLM-Assisted Retries] to improve LLM adherence to complex output schemas.
- [Observability] using OpenTelemetry, with native [Pydantic Logfire integration].
- [Type Annotations] to work nicely with linters and IDEs.
- [Configuration] options for multiple LLM providers including OpenAI, Anthropic, and Ollama.
- Many more features: [Chat Prompting], [Parallel Function Calling], [Vision], [Formatting], [Asyncio]...


## Installation

```sh
pip install magentic
```

or using uv

```sh
uv add magentic
```

Configure your OpenAI API key by setting the `OPENAI_API_KEY` environment variable. To configure a different LLM provider see [Configuration] for more.

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

See [Structured Outputs] for more.

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

See [Chat Prompting] for more.

### FunctionCall

An LLM can also decide to call functions. In this case the `@prompt`-decorated function returns a `FunctionCall` object which can be called to execute the function using the arguments provided by the LLM.

```python
from typing import Literal

from magentic import prompt, FunctionCall


def search_twitter(query: str, category: Literal["latest", "people"]) -> str:
    """Searches Twitter for a query."""
    print(f"Searching Twitter for {query!r} in category {category!r}")
    return "<twitter results>"


def search_youtube(query: str, channel: str = "all") -> str:
    """Searches YouTube for a query."""
    print(f"Searching YouTube for {query!r} in channel {channel!r}")
    return "<youtube results>"


@prompt(
    "Use the appropriate search function to answer: {question}",
    functions=[search_twitter, search_youtube],
)
def perform_search(question: str) -> FunctionCall[str]: ...


output = perform_search("What is the latest news on LLMs?")
print(output)
# > FunctionCall(<function search_twitter at 0x10c367d00>, 'LLMs', 'latest')
output()
# > Searching Twitter for 'Large Language Models news' in category 'latest'
# '<twitter results>'
```

See [Function Calling] for more.

### @prompt_chain

Sometimes the LLM requires making one or more function calls to generate a final answer. The `@prompt_chain` decorator will resolve `FunctionCall` objects automatically and pass the output back to the LLM to continue until the final answer is reached.

In the following example, when `describe_weather` is called the LLM first calls the `get_current_weather` function, then uses the result of this to formulate its final answer which gets returned.

```python
from magentic import prompt_chain


def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    # Pretend to query an API
    return {"temperature": "72", "forecast": ["sunny", "windy"]}


@prompt_chain(
    "What's the weather like in {city}?",
    functions=[get_current_weather],
)
def describe_weather(city: str) -> str: ...


describe_weather("Boston")
# 'The current weather in Boston is 72°F and it is sunny and windy.'
```

LLM-powered functions created using `@prompt`, `@chatprompt` and `@prompt_chain` can be supplied as `functions` to other `@prompt`/`@prompt_chain` decorators, just like regular python functions. This enables increasingly complex LLM-powered functionality, while allowing individual components to be tested and improved in isolation.

<!-- Links -->

[Structured Outputs]: structured-outputs.md
[Chat Prompting]: chat-prompting.md
[Function Calling]: function-calling.md
[Parallel Function Calling]: function-calling.md#parallelfunctioncall
[Observability]: logging-and-tracing.md
[Pydantic Logfire integration]: https://logfire.pydantic.dev/docs/integrations/third-party/magentic/
[Formatting]: formatting.md
[Asyncio]: asyncio.md
[Streaming]: streaming.md
[Vision]: vision.md
[LLM-assisted Retries]: retrying.md
[Configuration]: configuration.md
[Type Annotations]: type-checking.md
