# Function Calling

For many use cases, it is useful to provide the LLM with tools that it can choose when and how to use. In magentic this is done by passing a list of Python functions to the `functions` argument of a magentic decorator.

If the LLM chooses to call a function, the decorated function will return a `FunctionCall` instance. This object can be called to execute the function with the arguments that the LLM provided.

```python hl_lines="20"
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

## Choosing a specific tool

Set `tool_choice` to the name of one advertised tool while keeping other tools available:

```python
from magentic import Chat, FunctionCall, UserMessage, prompt


def search_web(query: str) -> str:
    return "web results"


def search_notes(query: str) -> str:
    return "note results"


@prompt(
    "Find information about {topic}",
    functions=[search_web, search_notes],
    tool_choice="search_notes",
)
def search(topic: str) -> FunctionCall[str]: ...


chat = Chat(
    [UserMessage("Find information about magentic")],
    functions=[search_web, search_notes],
    output_types=[FunctionCall],
)
chat = chat.submit(tool_choice="search_notes")
```

`@chatprompt` accepts the same option. Use `await chat.asubmit(tool_choice="search_notes")`
for async submission. The choice applies only to that submission; it is not stored on the returned
`Chat`. Decorated prompts retain their configured choice, including retries.

The selected name must match exactly one advertised tool. Function tools use their Python function
name. Structured-output tools use their generated schema name, such as `return_int` for `int`.
Unknown names and duplicate names, including a function/output-schema name collision, raise
`ValueError` before a request is sent. Selecting a function also requires an output type that allows
function calls. `Chat` defaults to `str`, so configure `output_types=[FunctionCall]` explicitly as above.
The choice does not change the output types, remove other tools, execute a function, or guarantee
that a remote model follows the request.

`ChatModel.complete` and `ChatModel.acomplete` keep their existing signatures. Direct model calls
use the additive `complete_with_tool_choice` and `acomplete_with_tool_choice` methods with a required
`tool_choice` name. Without the option, existing selection behavior is unchanged. Named selection
is implemented by the OpenAI, Anthropic, LiteLLM and OpenRouter adapters; the remote model/provider
must support it. The current Mistral adapter raises `NotImplementedError` for named selection.

Custom models opt in by implementing the new capability methods. Existing custom completion
overrides remain in use for ordinary requests. Inherited provider capabilities conservatively raise
`NotImplementedError` if a custom `complete` or `acomplete` override exists, even for a logging-only
wrapper. This avoids bypassing custom completion behavior. Such subclasses must explicitly implement
the corresponding capability if they want named selection. `@prompt_chain` does not accept this
option because forcing the same function on every turn can prevent the chain from finishing.

## FunctionCall

A `FunctionCall` combines a function with a set of arguments, ready to be called with no additional inputs required. In magentic, each time the LLM chooses to invoke a function a `FunctionCall` instance is returned. This allows the chosen function and supplied arguments to be validated or logged before the function is executed.

```python
from magentic import FunctionCall


def plus(a: int, b: int) -> int:
    return a + b


plus_1_2 = FunctionCall(plus, 1, b=2)
print(plus_1_2.function)
# > <function plus at 0x10c39cd30>
print(plus_1_2.arguments)
# > {'a': 1, 'b': 2}
plus_1_2()
# 3
```

## @prompt_chain

In some cases, you need the model to perform multiple function calls to reach a final answer. The `@prompt_chain` decorator is designed for this purpose. When a function decorated with `@prompt_chain` is called, the LLM is queried, then any function calls are automatically executed and the results appended to the list of messages. Then the LLM is queried again and this repeats until a final answer is reached.

In the following example, when `describe_weather` is called the LLM first calls the `get_current_weather` function, then uses the result of this to formulate its final answer which gets returned.

!!! tip
    The `@prompt_chain` decorator also accepts a sequence of `Message` objects as its first argument, similar to [`@chatprompt`](./chat-prompting.md).

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

LLM-powered functions created using `@prompt`, `@chatprompt` and `@prompt_chain` can be supplied as `functions` to other `@prompt`/`@prompt_chain` decorators, just like regular python functions!

To create a customized version of this function-execution loop, see [Chat#Agent](./chat.md#agent).

## ParallelFunctionCall

The most recent LLMs support "parallel function calling". This allows the model to call multiple functions at once. These functions can be executed concurrently, avoiding having to make several serial queries to the model.

You can use `ParallelFunctionCall` (and `AsyncParallelFunctionCall`) as a return annotation to indicate that you expect the LLM to make one or more function calls. The returned `ParallelFunctionCall` is a container of `FunctionCall` instances. When called, it returns a tuple of their results.

```python hl_lines="22"
from typing import Literal

from magentic import prompt, ParallelFunctionCall


def search_twitter(query: str, category: Literal["latest", "people"]) -> str:
    """Searches Twitter for a query."""
    print(f"Searching Twitter for {query!r} in category {category!r}")
    return "<twitter results>"


def search_youtube(query: str, channel: str = "all") -> str:
    """Searches YouTube for a query."""
    print(f"Searching YouTube for {query!r} in channel {channel!r}")
    return "<youtube results>"


@prompt(
    "Use the appropriate search functions to answer: {question}",
    functions=[search_twitter, search_youtube],
)
def perform_search(question: str) -> ParallelFunctionCall[str]: ...


output = perform_search("What is the latest news on LLMs?")
print(list(output))
# > [FunctionCall(<function search_twitter at 0x10c39f760>, 'LLMs', 'latest'),
#    FunctionCall(<function search_youtube at 0x10c39f7f0>, 'LLMs')]
output()
# > Searching Twitter for 'LLMs' in category 'latest'
# > Searching YouTube for 'LLMs' in channel 'all'
# ('<twitter results>', '<youtube results>')
```

## ParallelFunctionCall with @chatprompt

As with `FunctionCall` and Pydantic/Python objects, `ParallelFunctionCall` can be used with `@chatprompt` for few-shot prompting. In other words, to demonstrate to the LLM how/when it should use functions.

```python
from magentic import (
    chatprompt,
    AssistantMessage,
    FunctionCall,
    FunctionResultMessage,
    ParallelFunctionCall,
    UserMessage,
)


def plus(a: int, b: int) -> int:
    return a + b


def minus(a: int, b: int) -> int:
    return a - b


plus_1_2 = FunctionCall(plus, 1, 2)
minus_2_1 = FunctionCall(minus, 2, 1)


@chatprompt(
    UserMessage(
        "Sum 1 and 2. Also subtract 1 from 2.",
    ),
    AssistantMessage(ParallelFunctionCall([plus_1_2, minus_2_1])),
    FunctionResultMessage(3, plus_1_2),
    FunctionResultMessage(1, minus_2_1),
    UserMessage("Now add 4 to both results."),
    functions=[plus, minus],
)
def do_math() -> ParallelFunctionCall[int]: ...


output = do_math()
print(list(output))
# > [FunctionCall(<function plus at 0x10c3584c0>, 3, 4),
#    FunctionCall(<function plus at 0x10c3584c0>, 1, 4)]
output()
# (7, 5)
```

## Annotated Parameters

Like with `BaseModel`, you can use pydantic's `Field` to provide additional information for individual function parameters, such as a description. Here's how you could document for the model that the `temperature` parameter of the `activate_oven` function is measured in Fahrenheit and should be less than 500.

```python hl_lines="7"
from typing import Annotated, Literal

from pydantic import Field


def activate_oven(
    temperature: Annotated[int, Field(description="Temp in Fahrenheit", lt=500)],
    mode: Literal["broil", "bake", "roast"],
) -> str:
    """Turn the oven on with the provided settings."""
    return f"Preheating to {temperature} F with mode {mode}"
```

## ConfigDict

Also like with `BaseModel`, pydantic's (or magentic's) `ConfigDict` can be used with functions to configure behavior. Under the hood, the function gets converted to a pydantic model, with every function parameter becoming a field on that model. See the [Structured Outputs](structured-outputs.md) docs page for more information including the list of configuration options added by magentic.

```python hl_lines="3 7"
from typing import Annotated, Literal

from magentic import ConfigDict, with_config
from pydantic import Field


@with_config(ConfigDict(openai_strict=True))
def activate_oven(
    temperature: Annotated[int, Field(description="Temp in Fahrenheit", lt=500)],
    mode: Literal["broil", "bake", "roast"],
) -> str:
    """Turn the oven on with the provided settings."""
    return f"Preheating to {temperature} F with mode {mode}"
```
