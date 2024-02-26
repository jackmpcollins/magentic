# Chat Prompting

The `@chatprompt` decorator works just like `@prompt` but allows you to pass chat messages as a template rather than a single text prompt. This can be used to provide a system message or for few-shot prompting where you provide example responses to guide the model's output. Format fields denoted by curly braces `{example}` will be filled in all messages - use the `escape_braces` function to prevent a string being used as a template.

```python
from magentic import chatprompt, AssistantMessage, SystemMessage, UserMessage
from magentic.chatprompt import escape_braces

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
