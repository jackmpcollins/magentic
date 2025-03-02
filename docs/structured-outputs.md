# Structured Outputs

## Pydantic Models

The `@prompt` decorator will respect the return type annotation of the decorated function. This can be [any type supported by pydantic](https://docs.pydantic.dev/latest/usage/types/types/) including a `pydantic` model. See [the Pydantic docs](https://docs.pydantic.dev/latest/concepts/models/) for more information about models.

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

### Using `Field`

With pydantic's `BaseModel`, you can use `Field` to provide additional information for individual attributes, such as a description.

```python hl_lines="7-10"
from magentic import prompt
from pydantic import BaseModel, Field


class Superhero(BaseModel):
    name: str
    age: int = Field(
        description="The age of the hero, which could be much older than humans."
    )
    power: str = Field(examples=["Runs really fast"])
    enemies: list[str]


@prompt("Create a Superhero named {name}.")
def create_superhero(name: str) -> Superhero: ...
```

### ConfigDict

Pydantic also supports configuring the `BaseModel` by setting the `model_config` attribute. Magentic extends pydantic's `ConfigDict` class to add the following additional configuration options

- `openai_strict: bool` Indicates whether to use [OpenAI's Structured Outputs feature](https://platform.openai.com/docs/guides/structured-outputs/introduction).

See the [pydantic Configuration docs](https://docs.pydantic.dev/latest/api/config/) for the inherited configuration options.

```python hl_lines="1 6"
from magentic import prompt, ConfigDict
from pydantic import BaseModel


class Superhero(BaseModel):
    model_config = ConfigDict(openai_strict=True)

    name: str
    age: int
    power: str
    enemies: list[str]


@prompt("Create a Superhero named {name}.")
def create_superhero(name: str) -> Superhero: ...


create_superhero("Garden Man")
```

### JSON Schema

!!! note "OpenAI Structured Outputs"

    Setting `openai_strict=True` results in a different JSON schema than that from `.model_json_schema()` being sent to the LLM. Use `openai.pydantic_function_tool(Superhero)` to generate the JSON schema in this case.

You can generate the JSON schema for the pydantic model using the `.model_json_schema()` method. This is what is sent to the LLM.

Running `Superhero.model_json_schema()` for the above definition reveals the following JSON schema

```python
{
    "properties": {
        "name": {"title": "Name", "type": "string"},
        "age": {
            "description": "The age of the hero, which could be much older than humans.",
            "title": "Age",
            "type": "integer",
        },
        "power": {
            "examples": ["Runs really fast"],
            "title": "Power",
            "type": "string",
        },
        "enemies": {"items": {"type": "string"}, "title": "Enemies", "type": "array"},
    },
    "required": ["name", "age", "power", "enemies"],
    "title": "Superhero",
    "type": "object",
}
```

If a `StructuredOutputError` is raised often, this indicates that the LLM is failing to match the schema. The traceback for these errors includes the underlying pydantic `ValidationError` which shows in what way the received response was invalid. To combat these errors there are several options

- Add descriptions or examples for individual fields to demonstrate valid values.
- Simplify the output schema, including using more flexible types (e.g. `str` instead of `datetime`) or allowing fields to be nullable with `| None`.
- Switch to a "more intelligent" LLM. See [Configuration](configuration.md) for how to do this.

## Python Types

Regular Python types can also be used as the function return type.

```python
from magentic import prompt
from pydantic import BaseModel, Field


class Superhero(BaseModel):
    name: str
    age: int
    power: str
    enemies: list[str]


garden_man = Superhero(
    name="Garden Man",
    age=30,
    power="Control over plants",
    enemies=["Pollution Man", "Concrete Woman"],
)


@prompt("Return True if {hero.name} will be defeated by enemies {hero.enemies}")
def will_be_defeated(hero: Superhero) -> bool: ...


hero_defeated = will_be_defeated(garden_man)
print(hero_defeated)
# > True
```

## Chain-of-Thought Prompting

!!! warning "StreamedResponse"

    It is now recommended to use `StreamedResponse` for chain-of-thought prompting, as this uses the LLM provider's native chain-of-thought capabilities. See [StreamedResponse](streaming.md#streamedresponse) for more information.

Using a simple Python type as the return annotation might result in poor results as the LLM has no time to arrange its thoughts before answering. To allow the LLM to work through this "chain of thought" you can instead return a pydantic model with initial fields for explaining the final response.

```python hl_lines="5-9 20"
from magentic import prompt
from pydantic import BaseModel, Field


class ExplainedDefeated(BaseModel):
    explanation: str = Field(
        description="Describe the battle between the hero and their enemy."
    )
    defeated: bool = Field(description="True if the hero was defeated.")


class Superhero(BaseModel):
    name: str
    age: int
    power: str
    enemies: list[str]


@prompt("Return True if {hero.name} will be defeated by enemies {hero.enemies}")
def will_be_defeated(hero: Superhero) -> ExplainedDefeated: ...


garden_man = Superhero(
    name="Garden Man",
    age=30,
    power="Control over plants",
    enemies=["Pollution Man", "Concrete Woman"],
)

hero_defeated = will_be_defeated(garden_man)
print(hero_defeated.defeated)
# > True
print(hero_defeated.explanation)
# > 'Garden Man is an environmental hero who fights against Pollution Man ...'
```

### Explained

Using chain-of-thought is a common approach to improve the output of the model, so a generic `Explained` model might be generally useful. The `description` or `example` parameters of `Field` can be used to demonstrate the desired style and detail of the explanations.

```python
from typing import Generic, TypeVar

from magentic import prompt
from pydantic import BaseModel, Field


T = TypeVar("T")


class Explained(BaseModel, Generic[T]):
    explanation: str = Field(description="Explanation of how the value was determined.")
    value: T


@prompt("Return True if {hero.name} will be defeated by enemies {hero.enemies}")
def will_be_defeated(hero: Superhero) -> Explained[bool]: ...
```
