# Function Calling

## FunctionCall

A `FunctionCall` combines a function with a set of arguments, ready to be called.

```python
from magentic import FunctionCall


def plus(a: int, b: int) -> int:
    return a + b


plus_1_2 = FunctionCall(plus, 1, b=2)
plus_1_2()
# 3
```

The `@prompt` decorator has a `functions` parameter which provides the LLM with a list of Python functions that it can decide to call. If the LLM chooses to use a function, the decorated function will return a `FunctionCall` which you can then call to execute that function with the inputs provided by the LLM. This gives you the opportunity to validate the arguments that the LLM provided before the function is called.

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

## @prompt_chain

In some cases, you need the model to perform multiple function calls to reach a final answer. The `@prompt_chain` decorator will execute function calls automatically, append the result to the list of messages, and query the LLM again until a final answer is reached.

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

LLM-powered functions created using `@prompt` and `@prompt_chain` can be supplied as `functions` to other `@prompt`/`@prompt_chain` decorators, just like regular python functions!

## ParallelFunctionCall

The most recent LLMs support "parallel function calling". This allows the model to call multiple functions at once. These functions can be executed concurrently, avoiding having to make several serial queries to the model.

You can use `ParallelFunctionCall` (and `AsyncParallelFunctionCall`) as a return annotation to indicate that you expect the LLM to make one or more function calls. The returned `ParallelFunctionCall` is a container of `FunctionCall` instances. When called, it returns a tuple of their results.

```python
from magentic import prompt, ParallelFunctionCall


def plus(a: int, b: int) -> int:
    return a + b


def minus(a: int, b: int) -> int:
    return a - b


@prompt(
    "Sum {a} and {b}. Also subtract {a} from {b}.",
    functions=[plus, minus],
)
def plus_and_minus(a: int, b: int) -> ParallelFunctionCall[int]: ...


output = plus_and_minus(2, 3)
print(list(output))
# > [FunctionCall(<function plus at 0x106b8f010>, 2, 3), FunctionCall(<function minus at 0x106b8ef80>, 3, 2)]
output()
# (5, 1)
```

As with `FunctionCall` and other objects, `ParallelFunctionCall` can be used with `@chatprompt` for few-shot prompting. In other words, to demonstrate to the LLM how/when it should use functions.

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
# > [FunctionCall(<function plus at 0x10c3584c0>, 3, 4), FunctionCall(<function plus at 0x10c3584c0>, 1, 4)]
output()
# (7, 5)
```
