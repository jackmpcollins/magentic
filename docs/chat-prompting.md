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
def get_movie_quote(movie: str) -> Quote: ...


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
def get_similar_quote(movie: str, quote: Quote) -> Quote: ...


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


def change_music_volume(increment: int) -> int:
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
def adjust_for_guest(request: str) -> FunctionCall[None]: ...


func = adjust_for_guest("Do you have any more food?")
func()
# Ordered 3 pizza
```

## FunctionResultMessage

To include the result of calling the function in the messages use a `FunctionResultMessage`. This takes a `FunctionCall` instance as its second argument. The same `FunctionCall` instance must be passed to an `AssistantMessage` and the corresponding `FunctionResultMessage` so that the result can be correctly linked back to the function call that created it.

```python
from magentic import (
    chatprompt,
    AssistantMessage,
    FunctionCall,
    FunctionResultMessage,
    UserMessage,
)


def plus(a: int, b: int) -> int:
    return a + b


plus_1_2 = FunctionCall(plus, 1, 2)


@chatprompt(
    UserMessage("Use the plus function to add 1 and 2."),
    AssistantMessage(plus_1_2),
    FunctionResultMessage(3, plus_1_2),
    UserMessage("Now add 4 to the result."),
    functions=[plus],
)
def do_math() -> FunctionCall[int]: ...


do_math()
# FunctionCall(<function plus at 0x10a0829e0>, 3, 4)
```

## AnyMessage

The `AnyMessage` type can be used for (de)serialization of `Message` objects, or as a return type in prompt-functions. This allows you to create prompt-functions to do things like summarize a chat history into fewer messages, or even to create a set of messages that you can use in a chatprompt-function.

```python
from magentic import AnyMessage, prompt


@prompt("Create an example of few-shot prompting for a chatbot")
def make_few_shot_prompt() -> list[AnyMessage]: ...


make_few_shot_prompt()
# [SystemMessage('You are a helpful and knowledgeable assistant. You answer questions promptly and accurately. Always be polite and concise.'),
#  UserMessage('What’s the weather like today?'),
#  AssistantMessage[Any]('The weather today is sunny with a high of 75°F (24°C) and a low of 55°F (13°C). No chance of rain.'),
#  UserMessage('Can you explain the theory of relativity in simple terms?'),
#  AssistantMessage[Any]('Sure! The theory of relativity, developed by Albert Einstein, has two main parts: Special Relativity and General Relativity. Special Relativity is about how time and space are linked for objects moving at a consistent speed in a straight line. It shows that time can slow down or speed up depending on how fast you are moving compared to something else. General Relativity adds gravity into the mix and shows that massive objects cause space to bend and warp, which we feel as gravity.'),
#  UserMessage('How do I bake a chocolate cake?'),
#  AssistantMessage[Any]("Here's a simple recipe for a chocolate cake:\n\nIngredients:\n- 1 and 3/4 cups all-purpose flour\n- 1 and 1/2 cups granulated sugar\n- 3/4 cup cocoa powder\n- 1 and 1/2 teaspoons baking powder\n- 1 and 1/2 teaspoons baking soda\n- 1 teaspoon salt\n- 2 large eggs\n- 1 cup whole milk\n- 1/2 cup vegetable oil\n- 2 teaspoons vanilla extract\n- 1 cup boiling water\n\nInstructions:\n1. Preheat your oven to 350°F (175°C). Grease and flour two 9-inch round baking pans.\n2. In a large bowl, whisk together the flour, sugar, cocoa powder, baking powder, baking soda, and salt.\n3. Add the eggs, milk, oil, and vanilla. Beat on medium speed for 2 minutes.\n4. Stir in the boiling water (batter will be thin).\n5. Pour the batter evenly into the prepared pans.\n6. Bake for 30 to 35 minutes or until a toothpick inserted into the center comes out clean.\n7. Cool the cakes in the pans for 10 minutes, then remove them from the pans and cool completely on a wire rack.\n8. Frost with your favorite chocolate frosting and enjoy!")]
```

For (de)serialization, check out `TypeAdapter` from pydantic. See more on [the pydantic Type Adapter docs page](https://docs.pydantic.dev/latest/concepts/type_adapter/).

```python
from magentic import AnyMessage
from pydantic import TypeAdapter


messages = [
    {"role": "system", "content": "Hello"},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hello"},
    {"role": "tool", "content": 3, "tool_call_id": "unique_id"},
]
TypeAdapter(list[AnyMessage]).validate_python(messages)
# [SystemMessage('Hello'),
#  UserMessage('Hello'),
#  AssistantMessage[Any]('Hello'),
#  ToolResultMessage[Any](3, self.tool_call_id='unique_id')]
```
