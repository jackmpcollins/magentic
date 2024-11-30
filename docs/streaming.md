# Streaming

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

## Object Streaming

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

## StreamedResponse

Some LLMs have the ability to generate text output and make tool calls in the same response. This allows them to perform chain-of-thought reasoning or provide additional context to the user. In magentic, the `StreamedResponse` (or `AsyncStreamedResponse`) class can be used to request this type of output. This object is an iterable of `StreamedStr` (or `AsyncStreamedStr`) and `FunctionCall` instances.

!!! warning "Consuming StreamedStr"

    The StreamedStr object must be iterated over before the next item in the `StreamedResponse` is processed, otherwise the string output will be lost. This is because the `StreamedResponse` and `StreamedStr` share the same underlying generator, so advancing the `StreamedResponse` iterator skips over the `StreamedStr` items. The `StreamedStr` object has internal caching so after iterating over it once the chunks will remain available.

In the example below, we request that the LLM generates a greeting and then calls a function to get the weather for two cities. The `StreamedResponse` object is then iterated over to print the output, and the `StreamedStr` and `FunctionCall` items are processed separately.

```python
from magentic import prompt, FunctionCall, StreamedResponse, StreamedStr


def get_weather(city: str) -> str:
    return f"The weather in {city} is 20°C."


@prompt(
    "Say hello, then get the weather for: {cities}",
    functions=[get_weather],
)
def describe_weather(cities: list[str]) -> StreamedResponse: ...


response = describe_weather(["Cape Town", "San Francisco"])
for item in response:
    if isinstance(item, StreamedStr):
        for chunk in item:
            # print the chunks as they are received
            print(chunk, sep="", end="")
        print()
    if isinstance(item, FunctionCall):
        # print the function call, then call it and print the result
        print(item)
        print(item())

# Hello! I'll get the weather for Cape Town and San Francisco for you.
# FunctionCall(<function get_weather at 0x1109825c0>, 'Cape Town')
# The weather in Cape Town is 20°C.
# FunctionCall(<function get_weather at 0x1109825c0>, 'San Francisco')
# The weather in San Francisco is 20°C.
```
