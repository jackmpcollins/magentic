# Chat Prompting

## @chatprompt

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
def get_movie_quote(movie: str) -> Quote:
    ...


get_movie_quote("Iron Man")
# Quote(quote='I am Iron Man.', character='Tony Stark')
```

### escape_braces

To prevent curly braces from being interpreted as format fields, use the `escape_braces` function to escape them in strings.

```python
from magentic.chatprompt import escape_braces

string_with_braces = "Curly braces like {example} will be filled in!"
escaped_string = escape_braces(string_with_braces)
# 'Curly braces {{example}} will be filled in!'
escaped_string.format(example="test")
# 'Curly braces {example} will be filled in!'
```

## Placeholder

The `Placeholder` class enables templating of `AssistantMessage` content within the `@chatprompt` decorator. This allows dynamic changing of the messages used to prompt the model based on the arguments provided when the function is called.

```python
from magentic import chatprompt, AssistantMessage, Placeholder, UserMessage
from pydantic import BaseModel


class Quote(BaseModel):
    quote: str
    character: str


@chatprompt(
    UserMessage("Tell me a quote from {movie}"),
    AssistantMessage(Placeholder(Quote, "quote")),
    UserMessage("What is a similar quote from the same movie?"),
)
def get_similar_quote(movie: str, quote: Quote) -> Quote:
    ...


get_similar_quote(
    movie="Star Wars",
    quote=Quote(quote="I am your father", character="Darth Vader"),
)
# Quote(quote='The Force will be with you, always.', character='Obi-Wan Kenobi')
```

`Placeholder` can also be utilized in the `format` method of custom `Message` subclasses to provide an explicit way of inserting values from the function arguments. For example, see `UserImageMessage` in (TODO: link to GPT-vision page).

## FunctionCall

The content of an `AssistantMessage` can be a `FunctionCall`. This can be used to demonstrate to the LLM when/how it should call a function.

```python
from magentic import (
    chatprompt,
    AssistantMessage,
    FunctionCall,
    UserMessage,
    SystemMessage,
)


def change_music_volume(increment: int):
    """Change music volume level. Min 1, max 10."""
    print(f"Music volume change: {increment}")


def order_food(food: str, amount: int):
    """Order food."""
    print(f"Ordered {amount} {food}")


@chatprompt(
    SystemMessage(
        "You are hosting a party and must keep the guests happy."
        "Call functions as needed. Do not respond directly."
    ),
    UserMessage("It's pretty loud in here!"),
    AssistantMessage(FunctionCall(change_music_volume, -2)),
    UserMessage("{request}"),
    functions=[change_music_volume, order_food],
)
def adjust_for_guest(request: str) -> FunctionCall[None]:
    ...


func = adjust_for_guest("Do you have any more food?")
func()
# Ordered 3 pizza
```

To include the result of calling the function in the messages use a `FunctionResultMessage`.
