# magentic

Easily integrate Large Language Models into your Python code. Simply use the `@prompt` and `@chatprompt` decorators to create functions that return structured output from the LLM. Mix LLM queries and function calling with regular Python code to create complex logic.

## Features

- [Structured Outputs] using pydantic models and built-in python types.
- [Chat Prompting] to enable few-shot prompting with structured examples.
- [Function Calling] and [Parallel Function Calling] via the `FunctionCall` and `ParallelFunctionCall` return types.
- [Formatting] to naturally insert python objects into prompts.
- [Asyncio]. Simply use `async def` when defining a magentic function.
- [Streaming] structured outputs to use them as they are being generated.
- [Vision] to easily get stuctured outputs from images.
- Multiple LLM providers including OpenAI and Anthropic. See [Configuration].
- [Type Annotations] to work nicely with linters and IDEs.

## Installation

```sh
pip install magentic
```

or using poetry

```sh
poetry add magentic
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

<!-- Links -->

[Structured Outputs]: https://magentic.dev/structured-outputs
[Chat Prompting]: https://magentic.dev/chat-prompting
[Function Calling]: https://magentic.dev/function-calling
[Parallel Function Calling]: https://magentic.dev/function-calling/#parallelfunctioncall
[Formatting]: https://magentic.dev/formatting
[Asyncio]: https://magentic.dev/asyncio
[Streaming]: https://magentic.dev/streaming
[Vision]: https://magentic.dev/vision
[Configuration]: https://magentic.dev/configuration
[Type Annotations]: https://magentic.dev/type-checking

### Streaming

The `StreamedStr` (and `AsyncStreamedStr`) class can be used to stream the output of the LLM. This allows you to process the text while it is being generated, rather than receiving the whole output at once.

```python
from magentic import prompt, StreamedStr


@prompt("Tell me about {country}")
def describe_country(country: str) -> StreamedStr: ...


# Print the chunks while they are being received
for chunk in describe_country("Brazil"):
    print(chunk, end="")
# 'Brazil, officially known as the Federative Republic of Brazil, is ...'
```

Multiple `StreamedStr` can be created at the same time to stream LLM outputs concurrently. In the below example, generating the description for multiple countries takes approximately the same amount of time as for a single country.

```python
from time import time

countries = ["Australia", "Brazil", "Chile"]


# Generate the descriptions one at a time
start_time = time()
for country in countries:
    # Converting `StreamedStr` to `str` blocks until the LLM output is fully generated
    description = str(describe_country(country))
    print(f"{time() - start_time:.2f}s : {country} - {len(description)} chars")

# 22.72s : Australia - 2130 chars
# 41.63s : Brazil - 1884 chars
# 74.31s : Chile - 2968 chars


# Generate the descriptions concurrently by creating the StreamedStrs at the same time
start_time = time()
streamed_strs = [describe_country(country) for country in countries]
for country, streamed_str in zip(countries, streamed_strs):
    description = str(streamed_str)
    print(f"{time() - start_time:.2f}s : {country} - {len(description)} chars")

# 22.79s : Australia - 2147 chars
# 23.64s : Brazil - 2202 chars
# 24.67s : Chile - 2186 chars
```

### Object Streaming

Structured outputs can also be streamed from the LLM by using the return type annotation `Iterable` (or `AsyncIterable`). This allows each item to be processed while the next one is being generated.

```python
from collections.abc import Iterable
from time import time

from magentic import prompt
from pydantic import BaseModel


class Superhero(BaseModel):
    name: str
    age: int
    power: str
    enemies: list[str]


@prompt("Create a Superhero team named {name}.")
def create_superhero_team(name: str) -> Iterable[Superhero]: ...


start_time = time()
for hero in create_superhero_team("The Food Dudes"):
    print(f"{time() - start_time:.2f}s : {hero}")

# 2.23s : name='Pizza Man' age=30 power='Can shoot pizza slices from his hands' enemies=['The Hungry Horde', 'The Junk Food Gang']
# 4.03s : name='Captain Carrot' age=35 power='Super strength and agility from eating carrots' enemies=['The Sugar Squad', 'The Greasy Gang']
# 6.05s : name='Ice Cream Girl' age=25 power='Can create ice cream out of thin air' enemies=['The Hot Sauce Squad', 'The Healthy Eaters']
```

See [Streaming] for more.

### Asyncio

Asynchronous functions / coroutines can be used to concurrently query the LLM. This can greatly increase the overall speed of generation, and also allow other asynchronous code to run while waiting on LLM output. In the below example, the LLM generates a description for each US president while it is waiting on the next one in the list. Measuring the characters generated per second shows that this example achieves a 7x speedup over serial processing.

```python
import asyncio
from time import time
from typing import AsyncIterable

from magentic import prompt


@prompt("List ten presidents of the United States")
async def iter_presidents() -> AsyncIterable[str]: ...


@prompt("Tell me more about {topic}")
async def tell_me_more_about(topic: str) -> str: ...


# For each president listed, generate a description concurrently
start_time = time()
tasks = []
async for president in await iter_presidents():
    # Use asyncio.create_task to schedule the coroutine for execution before awaiting it
    # This way descriptions will start being generated while the list of presidents is still being generated
    task = asyncio.create_task(tell_me_more_about(president))
    tasks.append(task)

descriptions = await asyncio.gather(*tasks)

# Measure the characters per second
total_chars = sum(len(desc) for desc in descriptions)
time_elapsed = time() - start_time
print(total_chars, time_elapsed, total_chars / time_elapsed)
# 24575 28.70 856.07


# Measure the characters per second to describe a single president
start_time = time()
out = await tell_me_more_about("George Washington")
time_elapsed = time() - start_time
print(len(out), time_elapsed, len(out) / time_elapsed)
# 2206 18.72 117.78
```

See [Asyncio] for more.

### Additional Features

- The `functions` argument to `@prompt` can contain async/coroutine functions. When the corresponding `FunctionCall` objects are called the result must be awaited.
- The `Annotated` type annotation can be used to provide descriptions and other metadata for function parameters. See [the pydantic documentation on using `Field` to describe function arguments](https://docs.pydantic.dev/latest/usage/validation_decorator/#using-field-to-describe-function-arguments).
- The `@prompt` and `@prompt_chain` decorators also accept a `model` argument. You can pass an instance of `OpenaiChatModel` to use GPT4 or configure a different temperature. See below.
- Register other types to use as return type annotations in `@prompt` functions by following [the example notebook for a Pandas DataFrame](examples/custom_function_schemas/register_dataframe_function_schema.ipynb).

## Backend/LLM Configuration

Magentic supports multiple "backends" (LLM providers). These are

- `openai` : the default backend that uses the `openai` Python package. Supports all features of magentic.
  ```python
  from magentic import OpenaiChatModel
  ```
- `anthropic` : uses the `anthropic` Python package. Supports all features of magentic, however streaming responses are currently received all at once.
  ```sh
  pip install "magentic[anthropic]"
  ```
  ```python
  from magentic.chat_model.anthropic_chat_model import AnthropicChatModel
  ```
- `litellm` : uses the `litellm` Python package to enable querying LLMs from [many different providers](https://docs.litellm.ai/docs/providers). Note: some models may not support all features of `magentic` e.g. function calling/structured output and streaming.
  ```sh
  pip install "magentic[litellm]"
  ```
  ```python
  from magentic.chat_model.litellm_chat_model import LitellmChatModel
  ```
- `mistral` : uses the `openai` Python package with some small modifications to make the API queries compatible with the Mistral API. Supports all features of magentic, however tool calls (including structured outputs) are not streamed so are received all at once. Note: a future version of magentic might switch to using the `mistral` Python package.
  ```python
  from magentic.chat_model.mistral_chat_model import MistralChatModel
  ```

The backend and LLM (`ChatModel`) used by `magentic` can be configured in several ways. When a magentic function is called, the `ChatModel` to use follows this order of preference

1. The `ChatModel` instance provided as the `model` argument to the magentic decorator
1. The current chat model context, created using `with MyChatModel:`
1. The global `ChatModel` created from environment variables and the default settings in [src/magentic/settings.py](https://github.com/jackmpcollins/magentic/src/magentic/settings.py)

```python
from magentic import OpenaiChatModel, prompt
from magentic.chat_model.litellm_chat_model import LitellmChatModel


@prompt("Say hello")
def say_hello() -> str: ...


@prompt(
    "Say hello",
    model=LitellmChatModel("ollama_chat/llama3"),
)
def say_hello_litellm() -> str: ...


say_hello()  # Uses env vars or default settings

with OpenaiChatModel("gpt-3.5-turbo", temperature=1):
    say_hello()  # Uses openai with gpt-3.5-turbo and temperature=1 due to context manager
    say_hello_litellm()  # Uses litellm with ollama_chat/llama3 because explicitly configured
```

The following environment variables can be set.

| Environment Variable           | Description                              | Example                      |
| ------------------------------ | ---------------------------------------- | ---------------------------- |
| MAGENTIC_BACKEND               | The package to use as the LLM backend    | anthropic / openai / litellm |
| MAGENTIC_ANTHROPIC_MODEL       | Anthropic model                          | claude-3-haiku-20240307      |
| MAGENTIC_ANTHROPIC_API_KEY     | Anthropic API key to be used by magentic | sk-...                       |
| MAGENTIC_ANTHROPIC_BASE_URL    | Base URL for an Anthropic-compatible API | http://localhost:8080        |
| MAGENTIC_ANTHROPIC_MAX_TOKENS  | Max number of generated tokens           | 1024                         |
| MAGENTIC_ANTHROPIC_TEMPERATURE | Temperature                              | 0.5                          |
| MAGENTIC_LITELLM_MODEL         | LiteLLM model                            | claude-2                     |
| MAGENTIC_LITELLM_API_BASE      | The base url to query                    | http://localhost:11434       |
| MAGENTIC_LITELLM_MAX_TOKENS    | LiteLLM max number of generated tokens   | 1024                         |
| MAGENTIC_LITELLM_TEMPERATURE   | LiteLLM temperature                      | 0.5                          |
| MAGENTIC_MISTRAL_MODEL         | Mistral model                            | mistral-large-latest         |
| MAGENTIC_MISTRAL_API_KEY       | Mistral API key to be used by magentic   | XEG...                       |
| MAGENTIC_MISTRAL_BASE_URL      | Base URL for an Mistral-compatible API   | http://localhost:8080        |
| MAGENTIC_MISTRAL_MAX_TOKENS    | Max number of generated tokens           | 1024                         |
| MAGENTIC_MISTRAL_SEED          | Seed for deterministic sampling          | 42                           |
| MAGENTIC_MISTRAL_TEMPERATURE   | Temperature                              | 0.5                          |
| MAGENTIC_OPENAI_MODEL          | OpenAI model                             | gpt-4                        |
| MAGENTIC_OPENAI_API_KEY        | OpenAI API key to be used by magentic    | sk-...                       |
| MAGENTIC_OPENAI_API_TYPE       | Allowed options: "openai", "azure"       | azure                        |
| MAGENTIC_OPENAI_BASE_URL       | Base URL for an OpenAI-compatible API    | http://localhost:8080        |
| MAGENTIC_OPENAI_MAX_TOKENS     | OpenAI max number of generated tokens    | 1024                         |
| MAGENTIC_OPENAI_SEED           | Seed for deterministic sampling          | 42                           |
| MAGENTIC_OPENAI_TEMPERATURE    | OpenAI temperature                       | 0.5                          |

When using the `openai` backend, setting the `MAGENTIC_OPENAI_BASE_URL` environment variable or using `OpenaiChatModel(..., base_url="http://localhost:8080")` in code allows you to use `magentic` with any OpenAI-compatible API e.g. [Azure OpenAI Service](https://learn.microsoft.com/en-us/azure/ai-services/openai/quickstart?tabs=command-line&pivots=programming-language-python#create-a-new-python-application), [LiteLLM OpenAI Proxy Server](https://docs.litellm.ai/docs/proxy_server), [LocalAI](https://localai.io/howtos/easy-request-openai/). Note that if the API does not support tool calls then you will not be able to create prompt-functions that return Python objects, but other features of `magentic` will still work.

To use Azure with the openai backend you will need to set the `MAGENTIC_OPENAI_API_TYPE` environment variable to "azure" or use `OpenaiChatModel(..., api_type="azure")`, and also set the environment variables needed by the openai package to access Azure. See https://github.com/openai/openai-python#microsoft-azure-openai

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
   def random_color() -> str: ...
   ```
1. Use comment `# type: ignore[empty-body]` on each function. In this case you can add a docstring instead of `...`.
   ```python
   @prompt("Choose a color")
   def random_color() -> str:  # type: ignore[empty-body]
       """Returns a random color."""
   ```
