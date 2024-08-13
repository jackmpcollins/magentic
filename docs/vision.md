# Vision

Image inputs can be provided to LLMs in magentic by using the `UserImageMessage` message type.

!!! note "Anthropic Image URLs"

    Anthropic models currently do not support supplying an image as a url, just bytes.

For more information visit the [OpenAI Vision API documentation](https://platform.openai.com/docs/guides/vision) or the [Anthropic Vision API documentation](https://docs.anthropic.com/en/docs/build-with-claude/vision#example-multiple-images).

## UserImageMessage

The `UserImageMessage` can be used in `@chatprompt` alongside other messages. The LLM must be set to an OpenAI or Anthropic model that supports vision, for example `gpt-4o` (the default `ChatModel`). This can be done by passing the `model` parameter to `@chatprompt`, or through the other methods of [configuration](configuration.md).

```python
from pydantic import BaseModel, Field

from magentic import chatprompt, UserMessage
from magentic.vision import UserImageMessage


IMAGE_URL_WOODEN_BOARDWALK = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"


class ImageDetails(BaseModel):
    description: str = Field(description="A brief description of the image.")
    name: str = Field(description="A short name.")


@chatprompt(
    UserMessage("Describe the following image in one sentence."),
    UserImageMessage(IMAGE_URL_WOODEN_BOARDWALK),
)
def describe_image() -> ImageDetails: ...


image_details = describe_image()
print(image_details.name)
# 'Wooden Boardwalk in Green Wetland'
print(image_details.description)
# 'A serene wooden boardwalk meanders through a lush green wetland under a blue sky dotted with clouds.'
```

For more info on the `@chatprompt` decorator, see [Chat Prompting](chat-prompting.md).

## Placeholder

In the previous example, the image url was tied to the function. To provide the image as a function parameter, use `Placeholder`. This substitutes a function argument into the message when the function is called.

```python hl_lines="10"
from magentic import chatprompt, Placeholder, UserMessage
from magentic.vision import UserImageMessage


IMAGE_URL_WOODEN_BOARDWALK = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"


@chatprompt(
    UserMessage("Describe the following image in one sentence."),
    UserImageMessage(Placeholder(str, "image_url")),
)
def describe_image(image_url: str) -> str: ...


describe_image(IMAGE_URL_WOODEN_BOARDWALK)
# 'A wooden boardwalk meanders through lush green wetlands under a partly cloudy blue sky.'
```

## bytes

`UserImageMessage` can also accept `bytes` as input. Like `str`, this can be passed directly or via `Placeholder`.

```python
import requests

from magentic import chatprompt, Placeholder, UserMessage
from magentic.vision import UserImageMessage


IMAGE_URL_WOODEN_BOARDWALK = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"


def url_to_bytes(url: str) -> bytes:
    """Get the content of a URL as bytes."""

    # A custom user-agent is necessary to comply with Wikimedia user-agent policy
    # https://meta.wikimedia.org/wiki/User-Agent_policy
    headers = {"User-Agent": "MagenticExampleBot (https://magentic.dev/)"}
    return requests.get(url, headers=headers, timeout=10).content


@chatprompt(
    UserMessage("Describe the following image in one sentence."),
    UserImageMessage(Placeholder(bytes, "image_bytes")),
)
def describe_image(image_bytes: bytes) -> str: ...


image_bytes = url_to_bytes(IMAGE_URL_WOODEN_BOARDWALK)
describe_image(image_bytes)
# 'The image shows a wooden boardwalk extending through a lush green wetland with a backdrop of blue skies and scattered clouds.'
```
